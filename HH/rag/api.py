"""
HarnessHarvester - RAG HTTP API

Lightweight HTTP API server for RAG system access.
Built on stdlib http.server for zero-dependency operation.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict

from core.logging_setup import get_logger
from rag.management import RAGManager

logger = get_logger("rag_api")

# Module-level manager (lazy init)
_manager: RAGManager = None


def _get_manager() -> RAGManager:
    global _manager
    if _manager is None:
        _manager = RAGManager()
    return _manager


class RAGAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the RAG API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/api/health":
                self._respond(200, {"status": "ok", "service": "HarnessHarvester RAG API"})

            elif path == "/api/stats":
                mgr = _get_manager()
                self._respond(200, mgr.all_stats())

            elif path.startswith("/api/snippets"):
                self._handle_list("snippets", params)

            elif path.startswith("/api/errors"):
                self._handle_list("errors", params)

            elif path.startswith("/api/prompts"):
                self._handle_list("prompts", params)

            elif path == "/api/search":
                self._handle_search(params)

            else:
                self._respond(404, {"error": "Not found"})

        except Exception as e:
            logger.error(f"API error: {e}")
            self._respond(500, {"error": str(e)})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            body = self._read_body()

            if path == "/api/snippets":
                self._handle_add_snippet(body)

            elif path == "/api/errors":
                self._handle_add_error(body)

            elif path == "/api/prompts":
                self._handle_add_prompt(body)

            elif path == "/api/search":
                self._handle_search_post(body)

            elif path.endswith("/usage"):
                self._handle_record_usage(path, body)

            else:
                self._respond(404, {"error": "Not found"})

        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON body"})
        except Exception as e:
            logger.error(f"API error: {e}")
            self._respond(500, {"error": str(e)})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            parts = path.split("/")
            if len(parts) >= 4 and parts[1] == "api":
                store_type = parts[2]
                entry_id = parts[3]
                mgr = _get_manager()
                store = mgr.get_store(store_type)
                if hasattr(store, "delete_snippet"):
                    success = store.delete_snippet(entry_id)
                elif hasattr(store, "delete_error_pattern"):
                    success = store.delete_error_pattern(entry_id)
                elif hasattr(store, "delete_prompt"):
                    success = store.delete_prompt(entry_id)
                else:
                    success = store._engine.delete(entry_id)

                if success:
                    self._respond(200, {"deleted": entry_id})
                else:
                    self._respond(404, {"error": f"Entry {entry_id} not found"})
            else:
                self._respond(404, {"error": "Not found"})
        except Exception as e:
            logger.error(f"API error: {e}")
            self._respond(500, {"error": str(e)})

    # ── Handlers ─────────────────────────────────────────────

    def _handle_list(self, store_type: str, params: Dict):
        mgr = _get_manager()
        store = mgr.get_store(store_type)
        limit = int(params.get("limit", [50])[0])
        offset = int(params.get("offset", [0])[0])
        sort_by = params.get("sort_by", ["updated_at"])[0]

        if store_type in ("snippets", "snippet"):
            entries = store.list_snippets(limit=limit, offset=offset, sort_by=sort_by)
        elif store_type in ("errors", "error"):
            entries = store.list_patterns(limit=limit, offset=offset, sort_by=sort_by)
        else:
            entries = store.list_prompts(limit=limit, offset=offset, sort_by=sort_by)

        self._respond(200, {
            "entries": [e.to_dict() for e in entries],
            "count": len(entries),
            "offset": offset,
            "limit": limit,
        })

    def _handle_search(self, params: Dict):
        query = params.get("q", [""])[0]
        if not query:
            self._respond(400, {"error": "Missing 'q' parameter"})
            return

        top_k = int(params.get("top_k", [10])[0])
        store_type = params.get("store", [None])[0]
        mgr = _get_manager()

        if store_type:
            store = mgr.get_store(store_type)
            results = store.search(query, top_k=top_k) if hasattr(store, "search") else []
            self._respond(200, {
                "results": [
                    {
                        "entry": r.entry.to_dict(),
                        "score": r.final_score,
                        "scores": {
                            "tfidf": r.tfidf_score,
                            "bm25": r.bm25_score,
                            "quality": r.quality_score,
                            "usage": r.usage_score,
                            "recency": r.recency_score,
                        },
                    }
                    for r in results
                ],
                "query": query,
            })
        else:
            all_results = mgr.search_all(query, top_k=top_k)
            response = {"query": query}
            for key, results in all_results.items():
                response[key] = [
                    {"entry": r.entry.to_dict(), "score": r.final_score}
                    for r in results
                ]
            self._respond(200, response)

    def _handle_search_post(self, body: Dict):
        query = body.get("query", "")
        if not query:
            self._respond(400, {"error": "Missing 'query' field"})
            return
        top_k = body.get("top_k", 10)
        store_type = body.get("store", None)
        tags = body.get("tags", None)
        mgr = _get_manager()

        if store_type:
            store = mgr.get_store(store_type)
            results = store.search(query, top_k=top_k, tags=tags)
            self._respond(200, {
                "results": [
                    {"entry": r.entry.to_dict(), "score": r.final_score}
                    for r in results
                ],
                "query": query,
            })
        else:
            all_results = mgr.search_all(query, top_k=top_k, tags=tags)
            response = {"query": query}
            for key, results in all_results.items():
                response[key] = [
                    {"entry": r.entry.to_dict(), "score": r.final_score}
                    for r in results
                ]
            self._respond(200, response)

    def _handle_add_snippet(self, body: Dict):
        mgr = _get_manager()
        entry_id = mgr.snippets.add_snippet(
            content=body["content"],
            title=body.get("title", ""),
            description=body.get("description", ""),
            language=body.get("language", "python"),
            tags=body.get("tags"),
            source=body.get("source", "api"),
            quality_score=body.get("quality_score", 0.5),
        )
        self._respond(201, {"id": entry_id, "status": "created"})

    def _handle_add_error(self, body: Dict):
        mgr = _get_manager()
        entry_id = mgr.errors.add_error_pattern(
            error_message=body["error_message"],
            fix_description=body["fix_description"],
            fix_code=body.get("fix_code", ""),
            error_type=body.get("error_type", ""),
            traceback=body.get("traceback", ""),
            tags=body.get("tags"),
            source=body.get("source", "api"),
            quality_score=body.get("quality_score", 0.5),
        )
        self._respond(201, {"id": entry_id, "status": "created"})

    def _handle_add_prompt(self, body: Dict):
        mgr = _get_manager()
        entry_id = mgr.prompts.add_prompt_template(
            template=body["template"],
            task_type=body.get("task_type", ""),
            model_used=body.get("model_used", ""),
            description=body.get("description", ""),
            tags=body.get("tags"),
            source=body.get("source", "api"),
            quality_score=body.get("quality_score", 0.5),
        )
        self._respond(201, {"id": entry_id, "status": "created"})

    def _handle_record_usage(self, path: str, body: Dict):
        parts = path.split("/")
        if len(parts) >= 5:
            store_type = parts[2]
            entry_id = parts[3]
            success = body.get("success", True)
            mgr = _get_manager()
            store = mgr.get_store(store_type)
            if hasattr(store, "record_usage"):
                store.record_usage(entry_id, success=success)
            self._respond(200, {"recorded": True})
        else:
            self._respond(404, {"error": "Not found"})

    # ── Utility ──────────────────────────────────────────────

    def _read_body(self) -> Dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _respond(self, status: int, data: Any):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"API: {args[0] if args else ''}")


def start_api_server(host: str = "0.0.0.0", port: int = 8420):
    """Start the RAG API server."""
    server = HTTPServer((host, port), RAGAPIHandler)
    logger.info(f"RAG API server starting on {host}:{port}")
    print(f"RAG API server running at http://{host}:{port}")
    print("Endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/stats           - All store statistics")
    print("  GET  /api/search?q=...    - Search all stores")
    print("  GET  /api/snippets        - List snippets")
    print("  POST /api/snippets        - Add snippet")
    print("  GET  /api/errors          - List error patterns")
    print("  POST /api/errors          - Add error pattern")
    print("  GET  /api/prompts         - List prompt templates")
    print("  POST /api/prompts         - Add prompt template")
    print("  POST /api/search          - Search (POST with body)")
    print("  DELETE /api/{store}/{id}  - Delete entry")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        server.shutdown()
