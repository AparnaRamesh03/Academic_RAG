from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from state_shared import GraphState
from claim_verifier import verify_claims


def audit_answer(state: GraphState):
    """
    Claim-level answer auditor.
    """
    query = state["original_query"]
    answer = state.get("generation", "").strip()
    selected_docs = state.get("graded_docs", [])
    verify_retries = state.get("verify_retries", 0)

    print("\n[Final Combined] Auditing generated answer...")

    if not answer or not selected_docs:
        print("[Final Combined] Empty answer or no docs. Marking FAIL.")
        return {
            "citations_pass": False,
            "auditor_feedback": "The answer is empty or not grounded in retrieved evidence.",
            "verify_retries": verify_retries + 1,
            "claim_verification": [],
        }

    try:
        verification = verify_claims(
            query=query,
            answer=answer,
            docs=selected_docs,
        )
    except Exception as e:
        print("[Final Combined] Claim verification failed unexpectedly.")
        print(f"  -> Error: {e}")
        return {
            "citations_pass": False,
            "auditor_feedback": "Claim-level verification failed unexpectedly.",
            "verify_retries": verify_retries + 1,
            "claim_verification": [],
        }

    decision = verification.get("decision", "FAIL")
    overall_feedback = verification.get(
        "overall_feedback",
        "The answer contains unsupported or insufficiently grounded claims.",
    )
    claim_details = verification.get("claims", [])

    unsupported_claims = [c for c in claim_details if not c.get("supported", False)]

    if decision == "PASS" and not unsupported_claims:
        print("[Final Combined] Audit PASS.")
        return {
            "citations_pass": True,
            "auditor_feedback": "",
            "claim_verification": claim_details,
        }

    print("[Final Combined] Audit FAIL.")
    if unsupported_claims:
        print(f"  -> Unsupported claims: {len(unsupported_claims)}")
        for claim in unsupported_claims[:3]:
            print(f"     - {claim.get('claim_text', '')}")
            print(f"       feedback: {claim.get('feedback', '')}")

    return {
        "citations_pass": False,
        "auditor_feedback": overall_feedback,
        "verify_retries": verify_retries + 1,
        "claim_verification": claim_details,
    }