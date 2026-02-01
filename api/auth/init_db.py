from .db import engine, Base
from .models import User

print(">>> Creating database tables...")
Base.metadata.create_all(bind=engine)
print(">>> DB ready! Check auth.db in API folder.")
