from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    # Implement your authentication logic
    return {"user_id": "test-user", "is_admin": False}

async def require_role(role: str):
    def role_checker(user: dict = Depends(get_current_user)):
        if role == "admin" and not user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker