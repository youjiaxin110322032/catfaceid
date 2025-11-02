# libsql_db.py
import os

from dotenv import load_dotenv
load_dotenv()  # 讓本機可從 .env 讀 LIBSQL_URL / LIBSQL_AUTH_TOKEN

# --- 關鍵：使用「同步版」 client ---
try:
    # 方案 A：新版 libsql-client 提供 sync 介面
    from libsql_client.sync import create_client as create_client_sync
    _CREATE_SYNC = "pkg.sync.create_client"
except ImportError:
    try:
        # 方案 B：某些版本提供 create_client_sync
        from libsql_client import create_client_sync  # type: ignore
        _CREATE_SYNC = "pkg.create_client_sync"
    except Exception:
        # 方案 C：退而求其次，用 async 版本包一層同步（不建議，但可應急）
        from libsql_client import create_client as _create_client_async  # type: ignore
        import asyncio, aiohttp

        def create_client_sync(url: str, auth_token: str):
            # 在同步環境手動建立事件圈，避免 "no running event loop"
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # aiohttp 在 3.12 需要 running loop，所以把 client 建在 coroutine 裡
            async def _make():
                return _create_client_async({"url": url, "auth_token": auth_token})
            client = loop.run_until_complete(_make())
            return client
        _CREATE_SYNC = "wrapped_async_fallback"

LIBSQL_URL = os.getenv("LIBSQL_URL")
LIBSQL_AUTH_TOKEN = os.getenv("LIBSQL_AUTH_TOKEN")
if not LIBSQL_URL or not LIBSQL_AUTH_TOKEN:
    raise RuntimeError("請先設定環境變數 LIBSQL_URL / LIBSQL_AUTH_TOKEN")

# 真正建立同步 client
_client = create_client_sync(url=LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)
print(f"✅ libsql client ready via {_CREATE_SYNC}")

def query_all_cats():
    sql = """
    SELECT c.id, c.name, c.sex, c.coat, c.ear_tip,
           h.name AS household_name
    FROM cats c
    LEFT JOIN households h ON h.id = c.household_id
    ORDER BY c.id;
    """
    cats_rows = _client.execute(sql).rows

    care_sql = """
    SELECT ck.cat_id, t.name AS caretaker_name
    FROM cat_caretakers ck
    JOIN caretakers t ON t.id = ck.caretaker_id
    ORDER BY ck.cat_id;
    """
    care_rows = _client.execute(care_sql).rows

    care_map = {}
    for r in care_rows:
        cat_id = r["cat_id"]
        care_map.setdefault(cat_id, []).append(r["caretaker_name"])

    out = []
    for r in cats_rows:
        out.append({
            "id": r["id"],
            "name": r["name"],
            "sex": r["sex"],
            "coat": r["coat"],
            "ear_tip": bool(r["ear_tip"]),
            "household": r["household_name"],
            "caretakers": care_map.get(r["id"], []),
        })
    return out

def save_labels_to_db(id2name: dict):
    if not id2name:
        return
    for k, v in id2name.items():
        _client.execute(
            "INSERT OR REPLACE INTO cat_labels (id, name) VALUES (?, ?);",
            (int(k), v)
        )
