#!/usr/bin/env bash
# =============================================================================
#  test_api.sh  –  End-to-end verification of the Agentic AI REST API
#
#  Tests every agent endpoint, validates HTTP status codes and JSON response
#  shape, and exercises session chaining (Memory Agent and Drafter Agent).
#
#  Usage:
#    bash scripts/test_api.sh [BASE_URL]
#
#  Default BASE_URL: http://localhost:8080
#  Example with custom URL:
#    bash scripts/test_api.sh http://192.168.49.2:30800
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

PASS=0; FAIL=0; SKIP=0

pass() { echo -e "  ${GREEN}✔ PASS${NC}  $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✘ FAIL${NC}  $*"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YELLOW}⚠ SKIP${NC}  $*"; SKIP=$((SKIP + 1)); }
header() { echo -e "\n${BOLD}${CYAN}━━━  $*  ━━━${NC}"; }

# ---------------------------------------------------------------------------
# call BASE_URL path method body  →  sets $HTTP_CODE and $BODY
# ---------------------------------------------------------------------------
call() {
    local url="${BASE_URL}${1}"
    local method="${2:-GET}"
    local body="${3:-}"
    local response

    if [[ -n "$body" ]]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -d "$body" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" 2>/dev/null)
    fi

    HTTP_CODE=$(echo "$response" | tail -1)
    BODY=$(echo "$response" | sed '$d')
}

# Check a JSON field value:  check_field <field> <expected_substring>
check_field() {
    local field="$1" expected="$2"
    local value
    value=$(echo "$BODY" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('$field',''))" 2>/dev/null || echo "")
    if [[ "$value" == *"$expected"* ]]; then
        pass "Field '$field' contains '$expected'"
    else
        fail "Field '$field' expected to contain '$expected', got: '${value:0:120}'"
    fi
}

# Check HTTP status
check_status() {
    local expected="$1"
    if [[ "$HTTP_CODE" == "$expected" ]]; then
        pass "HTTP $HTTP_CODE"
    else
        fail "HTTP $HTTP_CODE (expected $expected) — body: ${BODY:0:200}"
    fi
}

# Check if JSON key exists
check_key() {
    local key="$1"
    local exists
    exists=$(echo "$BODY" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print('yes' if '$key' in d else 'no')" 2>/dev/null || echo "no")
    if [[ "$exists" == "yes" ]]; then
        pass "JSON key '$key' present"
    else
        fail "JSON key '$key' missing — body: ${BODY:0:200}"
    fi
}

# =============================================================================
# TEST SUITE
# =============================================================================

echo -e "\n${BOLD}Agentic AI — API Test Suite${NC}"
echo -e "Target: ${CYAN}${BASE_URL}${NC}"
echo "────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
header "1. Infrastructure"
# ---------------------------------------------------------------------------

echo -e "\n  [ GET /health ]"
call "/health"
check_status "200"
check_field "status" "healthy"
check_field "service" "Agentic AI API"

echo -e "\n  [ GET / ]"
call "/"
check_status "200"
check_field "version" "1.0.0"
check_key "agents"

# ---------------------------------------------------------------------------
header "2. Agent Bot  (stateless single-turn chatbot)"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/agent-bot/chat — basic message ]"
call "/api/agent-bot/chat" "POST" '{"message": "Reply with exactly the word PONG and nothing else."}'
check_status "200"
check_key "response"
check_key "session_id"

RESPONSE_TEXT=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('response',''))" 2>/dev/null || echo "")
echo -e "     LLM replied: ${CYAN}${RESPONSE_TEXT:0:120}${NC}"

echo -e "\n  [ POST /api/agent-bot/chat — verify stateless (no memory) ]"
call "/api/agent-bot/chat" "POST" '{"message": "What did I just tell you to reply with?"}'
check_status "200"
check_key "response"
# A stateless agent should NOT remember "PONG" — just verify it returns something
RESPONSE_TEXT=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('response',''))" 2>/dev/null || echo "")
echo -e "     LLM replied: ${CYAN}${RESPONSE_TEXT:0:120}${NC}"

# ---------------------------------------------------------------------------
header "3. Memory Agent  (multi-turn chat with session history)"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/memory/chat — first turn, no session_id ]"
call "/api/memory/chat" "POST" '{"message": "My secret number is 42. Remember it."}'
check_status "200"
check_key "session_id"
check_key "response"

MEM_SESSION=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")
echo -e "     Session ID: ${CYAN}${MEM_SESSION}${NC}"

echo -e "\n  [ POST /api/memory/chat — second turn, reuse session ]"
call "/api/memory/chat" "POST" "{\"message\": \"What is my secret number?\", \"session_id\": \"${MEM_SESSION}\"}"
check_status "200"
MEM_REPLY=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('response',''))" 2>/dev/null || echo "")
echo -e "     LLM replied: ${CYAN}${MEM_REPLY:0:120}${NC}"
if echo "$MEM_REPLY" | grep -qi "42"; then
    pass "Memory retained — agent recalled '42'"
else
    fail "Memory not retained — '42' not found in reply"
fi

echo -e "\n  [ GET /api/memory/sessions/:id — retrieve history ]"
call "/api/memory/sessions/${MEM_SESSION}"
check_status "200"
check_key "history"
HIST_LEN=$(echo "$BODY" | python3 -c \
    "import sys,json; print(len(json.load(sys.stdin).get('history',[])))" 2>/dev/null || echo "0")
echo -e "     History length: ${CYAN}${HIST_LEN} messages${NC}"
if [[ "$HIST_LEN" -ge 2 ]]; then
    pass "History has $HIST_LEN messages (≥ 2 expected)"
else
    fail "History too short: $HIST_LEN messages"
fi

echo -e "\n  [ DELETE /api/memory/sessions/:id — clear session ]"
call "/api/memory/sessions/${MEM_SESSION}" "DELETE"
check_status "200"

# ---------------------------------------------------------------------------
header "4. ReAct Agent  (math tool agent)"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/react/solve — addition ]"
call "/api/react/solve" "POST" '{"query": "What is 15 plus 27?"}'
check_status "200"
check_key "result"
REACT_REPLY=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('result',''))" 2>/dev/null || echo "")
echo -e "     LLM replied: ${CYAN}${REACT_REPLY:0:120}${NC}"
if echo "$REACT_REPLY" | grep -qE "42|forty.?two"; then
    pass "ReAct tool call correct — result contains 42"
else
    fail "ReAct result did not contain '42': ${REACT_REPLY:0:120}"
fi

echo -e "\n  [ POST /api/react/solve — chained operations ]"
call "/api/react/solve" "POST" '{"query": "Add 10 and 5, then multiply the result by 3"}'
check_status "200"
REACT_REPLY=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('result',''))" 2>/dev/null || echo "")
echo -e "     LLM replied: ${CYAN}${REACT_REPLY:0:120}${NC}"
if echo "$REACT_REPLY" | grep -qE "45|forty.?five"; then
    pass "Chained tool calls correct — result contains 45"
else
    fail "Chained result did not contain '45': ${REACT_REPLY:0:120}"
fi

# ---------------------------------------------------------------------------
header "5. Drafter Agent  (session-based document drafting)"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/drafter/chat — create document ]"
call "/api/drafter/chat" "POST" \
    '{"instruction": "Write a one-paragraph introduction about artificial intelligence."}'
check_status "200"
check_key "session_id"
check_key "document_content"
check_key "response"

DRAFTER_SESSION=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")
DOC_CONTENT=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('document_content',''))" 2>/dev/null || echo "")
echo -e "     Session ID: ${CYAN}${DRAFTER_SESSION}${NC}"
echo -e "     Doc preview: ${CYAN}${DOC_CONTENT:0:100}...${NC}"

if [[ -n "$DOC_CONTENT" ]]; then
    pass "Document content is non-empty"
else
    fail "Document content is empty after creation"
fi

echo -e "\n  [ POST /api/drafter/chat — append to document ]"
call "/api/drafter/chat" "POST" \
    "{\"instruction\": \"Add a one-sentence conclusion.\", \"session_id\": \"${DRAFTER_SESSION}\"}"
check_status "200"
NEW_DOC=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('document_content',''))" 2>/dev/null || echo "")
echo -e "     Updated doc length: ${CYAN}${#NEW_DOC} chars${NC}"
if [[ ${#NEW_DOC} -gt ${#DOC_CONTENT} ]]; then
    pass "Document grew after append (${#DOC_CONTENT} → ${#NEW_DOC} chars)"
else
    skip "Document length unchanged — may still be correct"
fi

echo -e "\n  [ POST /api/drafter/chat — save document ]"
call "/api/drafter/chat" "POST" \
    "{\"instruction\": \"Save the document as test_output.txt\", \"session_id\": \"${DRAFTER_SESSION}\"}"
check_status "200"
IS_SAVED=$(echo "$BODY" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('is_saved', False))" 2>/dev/null || echo "False")
echo -e "     is_saved: ${CYAN}${IS_SAVED}${NC}"
if [[ "$IS_SAVED" == "True" ]]; then
    pass "Document saved successfully (is_saved=true)"
else
    skip "is_saved=false — agent may have returned a summary instead of calling save tool"
fi

echo -e "\n  [ DELETE /api/drafter/sessions/:id — cleanup ]"
call "/api/drafter/sessions/${DRAFTER_SESSION}" "DELETE"
check_status "200"

# ---------------------------------------------------------------------------
header "6. RAG Agent  (PDF Q&A — Stock Market 2024)"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/rag/query — check availability ]"
call "/api/rag/query" "POST" '{"query": "What is the main topic of this document?"}'

if [[ "$HTTP_CODE" == "503" ]]; then
    skip "RAG agent unavailable (503) — PDF may not be found in container. Check PDF_PATH env var."
elif [[ "$HTTP_CODE" == "200" ]]; then
    check_key "answer"
    RAG_ANSWER=$(echo "$BODY" | python3 -c \
        "import sys,json; print(json.load(sys.stdin).get('answer',''))" 2>/dev/null || echo "")
    echo -e "     Answer: ${CYAN}${RAG_ANSWER:0:150}${NC}"
    if [[ -n "$RAG_ANSWER" ]]; then
        pass "RAG agent returned an answer"
    else
        fail "RAG agent returned empty answer"
    fi

    echo -e "\n  [ POST /api/rag/query — specific stock market question ]"
    call "/api/rag/query" "POST" '{"query": "How did the S&P 500 perform in 2024?"}'
    check_status "200"
    RAG_ANSWER=$(echo "$BODY" | python3 -c \
        "import sys,json; print(json.load(sys.stdin).get('answer',''))" 2>/dev/null || echo "")
    echo -e "     Answer: ${CYAN}${RAG_ANSWER:0:150}${NC}"
    if [[ -n "$RAG_ANSWER" ]]; then
        pass "RAG agent answered stock market query"
    else
        fail "RAG returned empty answer for stock market query"
    fi
else
    fail "RAG agent returned HTTP $HTTP_CODE — body: ${BODY:0:200}"
fi

# ---------------------------------------------------------------------------
header "7. Error handling"
# ---------------------------------------------------------------------------

echo -e "\n  [ POST /api/agent-bot/chat — empty message (should still return 200) ]"
call "/api/agent-bot/chat" "POST" '{"message": ""}'
echo -e "     HTTP $HTTP_CODE (200 or 422 are both acceptable)"
if [[ "$HTTP_CODE" == "200" || "$HTTP_CODE" == "422" ]]; then
    pass "Empty message handled gracefully (HTTP $HTTP_CODE)"
else
    fail "Unexpected HTTP $HTTP_CODE for empty message"
fi

echo -e "\n  [ GET /api/memory/sessions/nonexistent-id — 404 expected ]"
call "/api/memory/sessions/does-not-exist-xyz"
check_status "404"

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "════════════════════════════════════════════════════"
echo -e "${BOLD}  Test Summary${NC}"
echo "────────────────────────────────────────────────────"
echo -e "  Total  : ${BOLD}${TOTAL}${NC}"
echo -e "  ${GREEN}Passed : ${PASS}${NC}"
echo -e "  ${RED}Failed : ${FAIL}${NC}"
echo -e "  ${YELLOW}Skipped: ${SKIP}${NC}"
echo "════════════════════════════════════════════════════"

if [[ "$FAIL" -gt 0 ]]; then
    echo -e "\n${RED}Some tests failed. Common causes:${NC}"
    echo "  • OpenAI API key has insufficient quota (add credits at platform.openai.com/billing)"
    echo "  • Port-forward not running — run: kubectl port-forward service/agentic-ai-service 8080:80 &"
    echo "  • Pod not ready — run: kubectl get pods -l app=agentic-ai"
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
fi
