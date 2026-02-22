import psycopg2

try:
    conn = psycopg2.connect(
        host="db.trwsfdhxzwzkjandsmvz.supabase.co",
        database="postgres",
        user="postgres",
        password="RkoGkPLWxh4vavX3",
        port=5432
    )
    print("✅ DB CONNECTED")
    conn.close()
except Exception as e:
    print("❌ DB FAILED:", e)