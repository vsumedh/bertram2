#!/bin/bash
#
# compare_demo_logs.sh - Validate demo mode log consistency
#
# This script runs a fixed episode and checks that:
# 1. Dialogue logs (GREEN/WHITE) are formatted consistently
# 2. Evaluation logs (JUDGE/EXPERT) match the same visual style
# 3. All required sections are present
#
# Usage:
#   ./scripts/compare_demo_logs.sh [task_number]
#
# Example:
#   ./scripts/compare_demo_logs.sh 4

set -e

TASK_NUM="${1:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/demo_log_$$_task${TASK_NUM}.txt"

echo "=================================================="
echo "Demo Mode Log Consistency Check"
echo "=================================================="
echo "Task: $TASK_NUM"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_DIR"

# Run evaluation and capture output
echo "Running evaluation..."
DEMO_MODE=1 timeout 180 python main.py evaluate --tasks "$TASK_NUM" --agent llm 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "Checking Log Structure"
echo "=================================================="

# Strip ANSI codes from log for pattern matching
CLEAN_LOG="/tmp/demo_log_clean_$$_task${TASK_NUM}.txt"
sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" > "$CLEAN_LOG"

# Function to check for pattern and report
check_pattern() {
    local pattern="$1"
    local description="$2"
    if grep -q "$pattern" "$CLEAN_LOG"; then
        echo "✓ $description"
        return 0
    else
        echo "✗ $description - MISSING"
        return 1
    fi
}

ERRORS=0

# Check dialogue structure
echo ""
echo "Dialogue Structure:"
check_pattern "^Step [0-9]" "Step numbers present" || ((ERRORS++))
check_pattern "GREEN" "GREEN agent identifier" || ((ERRORS++))
check_pattern "WHITE" "WHITE agent identifier" || ((ERRORS++))
check_pattern "Observation:" "Observation labels" || ((ERRORS++))
check_pattern "Command:" "Command labels" || ((ERRORS++))
check_pattern "Reasoning:" "Reasoning labels" || ((ERRORS++))

# Check evaluation structure
echo ""
echo "Evaluation Structure:"
check_pattern "═" "Double dividers for major sections" || ((ERRORS++))
check_pattern "EVALUATION" "EVALUATION header" || ((ERRORS++))
check_pattern "JUDGE" "JUDGE agent identifier" || ((ERRORS++))
check_pattern "EXPERT" "EXPERT section" || ((ERRORS++))
check_pattern "OVERALL" "OVERALL score section" || ((ERRORS++))

# Check score components
echo ""
echo "Score Components:"
check_pattern "Correctness" "Correctness score" || ((ERRORS++))
check_pattern "Efficiency" "Efficiency score" || ((ERRORS++))
check_pattern "Strategy" "Strategy score" || ((ERRORS++))
check_pattern "Reasoning" "Reasoning score" || ((ERRORS++))
check_pattern "/ 10" "Score denominators" || ((ERRORS++))

# Check visual elements
echo ""
echo "Visual Elements:"
check_pattern "✓\|✗" "Status indicators (checkmark/X)" || ((ERRORS++))
check_pattern "SUCCESS\|FAILED" "Outcome text" || ((ERRORS++))

# Check expert actions
echo ""
echo "Expert Trajectory:"
check_pattern "Actions:" "Expert actions list" || ((ERRORS++))
check_pattern "1\. " "Numbered action list" || ((ERRORS++))

# Check weight calculation
echo ""
echo "Score Calculation:"
check_pattern "correctness×0.30" "Weight breakdown" || ((ERRORS++))

echo ""
echo "=================================================="
echo "Consistency Checks"
echo "=================================================="

# Check that ANSI codes are balanced
BOLD_OPENS=$(grep -o '\[1m' "$LOG_FILE" | wc -l)
RESETS=$(grep -o '\[0m' "$LOG_FILE" | wc -l)
if [ "$BOLD_OPENS" -le "$RESETS" ]; then
    echo "✓ ANSI codes balanced ($BOLD_OPENS opens, $RESETS resets)"
else
    echo "⚠ ANSI codes may be unbalanced ($BOLD_OPENS opens, $RESETS resets)"
fi

# Check that dividers are consistent length
DIV_COUNT=$(grep -E '^[─═┄]{70,}$' "$LOG_FILE" | wc -l)
echo "✓ Found $DIV_COUNT divider lines"

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="

if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
    rm -f "$LOG_FILE" "$CLEAN_LOG"
    exit 0
else
    echo "✗ $ERRORS check(s) failed"
    echo "  Log saved to: $LOG_FILE"
    rm -f "$CLEAN_LOG"
    exit 1
fi

