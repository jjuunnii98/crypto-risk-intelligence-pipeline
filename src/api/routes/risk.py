from fastapi import APIRouter

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/latest")
def get_latest_risk():
    return {"message": "latest risk endpoint placeholder"}