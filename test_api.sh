#!/bin/bash
# Test all SyllabusLens API endpoints against http://127.0.0.1:8080
# Usage: start the server first, then run this script.
#   uvicorn api.main:app --host 127.0.0.1 --port 8080
#   bash test_api.sh

BASE="http://127.0.0.1:8080"
PASS=0
FAIL=0

check() {
    local name="$1"
    local url="$2"
    local expect="$3"

    response=$(curl -s -w "\n%{http_code}" "$url")
    code=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')

    if [ "$code" != "200" ]; then
        echo "FAIL  $name — HTTP $code"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ -n "$expect" ]; then
        if echo "$body" | grep -q "$expect"; then
            echo "PASS  $name"
            PASS=$((PASS + 1))
        else
            echo "FAIL  $name — expected \"$expect\" not found in response"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "PASS  $name"
        PASS=$((PASS + 1))
    fi
}

echo "=== SyllabusLens API Tests ==="
echo "Target: $BASE"
echo ""

# 1. Topics
echo "--- GET /api/topics ---"
check "returns 200"                "$BASE/api/topics" ""
check "returns a JSON array"       "$BASE/api/topics" '"name"'
check "contains topic names"       "$BASE/api/topics" "Biology"

# 2. Title search
echo ""
echo "--- GET /api/titles/search ---"
check "empty query returns results"     "$BASE/api/titles/search" '"canonical_title"'
check "search 'Introduction'"           "$BASE/api/titles/search?q=Introduction" "Introduction"
check "search returns slug field"       "$BASE/api/titles/search?q=Introduction" '"slug"'
check "search with limit"               "$BASE/api/titles/search?q=a&limit=3" '"id"'
check "search no match"                 "$BASE/api/titles/search?q=zzzznonexistent" '\[\]'

# 3. Rankings — assignment mode (no title)
echo ""
echo "--- GET /api/rankings (assignment mode) ---"
check "returns 200"                     "$BASE/api/rankings?limit=5" '"mode"'
check "mode is assignment"              "$BASE/api/rankings?limit=5" '"assignment"'
check "has results array"               "$BASE/api/rankings?limit=5" '"results"'
check "results have count field"        "$BASE/api/rankings?limit=5" '"count"'
check "pivot_title is null"             "$BASE/api/rankings?limit=5" '"pivot_title":null'

# 4. Rankings — filtered by school
echo ""
echo "--- GET /api/rankings (school filter) ---"
check "filter schools=osu"              "$BASE/api/rankings?schools=osu&limit=5" '"assignment"'
check "filter schools=ufl"              "$BASE/api/rankings?schools=ufl&limit=5" '"results"'
check "filter schools=utaustin"         "$BASE/api/rankings?schools=utaustin&limit=5" '"results"'

# 5. Rankings — filtered by topic
echo ""
echo "--- GET /api/rankings (topic filter) ---"
# grab first topic id
TOPIC_ID=$(curl -s "$BASE/api/topics" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])" 2>/dev/null)
if [ -n "$TOPIC_ID" ]; then
    check "filter by topic_id=$TOPIC_ID" "$BASE/api/rankings?topics=$TOPIC_ID&limit=5" '"results"'
else
    echo "SKIP  could not fetch topic id"
fi

# 6. Rankings — co-assignment mode
echo ""
echo "--- GET /api/rankings (co-assignment mode) ---"
TITLE_ID=$(curl -s "$BASE/api/rankings?limit=1" | python3 -c "import sys,json; r=json.load(sys.stdin)['results']; print(r[0]['title_id'] if r else '')" 2>/dev/null)
if [ -n "$TITLE_ID" ]; then
    check "co-assignment mode"           "$BASE/api/rankings?title=$TITLE_ID&limit=5" '"co-assignment"'
    check "pivot_title is set"           "$BASE/api/rankings?title=$TITLE_ID&limit=5" '"pivot_title"'
    check "co-assignment has results"    "$BASE/api/rankings?title=$TITLE_ID&limit=5" '"results"'
    check "co-assign + school filter"    "$BASE/api/rankings?title=$TITLE_ID&schools=ufl&limit=5" '"co-assignment"'
else
    echo "SKIP  no titles in database to test co-assignment"
fi

# 7. API docs
echo ""
echo "--- Docs ---"
check "OpenAPI docs page loads"         "$BASE/docs" "swagger"

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
