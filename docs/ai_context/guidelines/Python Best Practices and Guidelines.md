# Python Best Practices & Guidelines (2026 Edition)

## A Comprehensive Guide for Writing Maintainable, High-Quality Python Code

---

## Table of Contents

1. [Philosophy & Guiding Principles](#1-philosophy--guiding-principles)
2. [SOLID Principles in Python](#2-solid-principles-in-python)
3. [Additional Design Principles (DRY, KISS, YAGNI, GRASP)](#3-additional-design-principles)
4. [GoF Design Patterns in Python](#4-gof-design-patterns-in-python)
5. [Clean Code Practices](#5-clean-code-practices)
6. [Modern Python Features (3.10–3.13)](#6-modern-python-features-310313)
7. [Type System & Generics](#7-type-system--generics)
8. [Data Structures & Collections](#8-data-structures--collections)
9. [Module Organization & Avoiding Cyclic Dependencies](#9-module-organization--avoiding-cyclic-dependencies)
10. [Error Handling & Exceptions](#10-error-handling--exceptions)
11. [Testing Best Practices](#11-testing-best-practices)
12. [Project Structure & Tooling](#12-project-structure--tooling)
13. [Performance Considerations](#13-performance-considerations)
14. [Quick Reference Checklists](#14-quick-reference-checklists)

---

## 1. Philosophy & Guiding Principles

### 1.1 The Zen of Python (PEP 20)

Run `import this` in any Python interpreter to see these guiding principles. The most actionable ones:

| Principle                                    | Practical Meaning                                          |
| -------------------------------------------- | ---------------------------------------------------------- |
| **Beautiful is better than ugly**            | Code is read 10x more than written. Invest in readability. |
| **Explicit is better than implicit**         | Don't hide behavior. Make intentions clear.                |
| **Simple is better than complex**            | Choose the straightforward solution first.                 |
| **Flat is better than nested**               | Avoid deep nesting; use early returns and guard clauses.   |
| **Readability counts**                       | Use clear names, proper spacing, and logical structure.    |
| **Errors should never pass silently**        | Handle exceptions explicitly; don't suppress them.         |
| **There should be one obvious way to do it** | Follow established patterns and conventions.               |

### 1.2 The Core Question

Before writing any code, ask yourself:

> **"Can another developer understand this code in 6 months without my help?"**

If the answer is "no," simplify.

---

## 2. SOLID Principles in Python

SOLID principles prevent rigid, fragile, and hard-to-change code. Here's how each applies specifically to Python.

### 2.1 Single Responsibility Principle (SRP)

> **Rule:** A class or function should have exactly one reason to change.

**Why it matters:**
- Easier to test (one behavior = one test suite)
- Easier to understand (focused purpose)
- Changes in one area don't break unrelated functionality

**❌ DON'T: A "God Class" doing everything**
```python
class UserManager:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_user(self, data):
        # Validates data
        if not data.get("email"):
            raise ValueError("Email required")
        # Saves to database
        self.db.execute("INSERT INTO users ...")
        # Sends welcome email
        self._send_email(data["email"], "Welcome!")
        # Logs the action
        print(f"User {data['email']} created")
        # Updates analytics
        self._track_event("user_created")
    
    def _send_email(self, to, subject):
        # SMTP logic here...
        pass
    
    def _track_event(self, event_name):
        # Analytics logic here...
        pass
```

**✅ DO: Separate responsibilities into focused classes**
```python
from dataclasses import dataclass
from typing import Protocol

# Each class has ONE job

class UserValidator:
    """Validates user data."""
    def validate(self, data: dict) -> None:
        if not data.get("email"):
            raise ValueError("Email required")
        if "@" not in data["email"]:
            raise ValueError("Invalid email format")

class UserRepository:
    """Handles user persistence."""
    def __init__(self, db_connection):
        self.db = db_connection
    
    def save(self, user_data: dict) -> int:
        # Returns user ID
        return self.db.execute("INSERT INTO users ...")

class EmailService:
    """Sends emails."""
    def send_welcome(self, email: str) -> None:
        # SMTP logic here
        pass

class AnalyticsService:
    """Tracks events."""
    def track(self, event_name: str, metadata: dict = None) -> None:
        pass

# Coordinator class that composes the others
class UserRegistrationService:
    """Coordinates user registration workflow."""
    def __init__(
        self,
        validator: UserValidator,
        repository: UserRepository,
        email_service: EmailService,
        analytics: AnalyticsService,
    ):
        self._validator = validator
        self._repository = repository
        self._email = email_service
        self._analytics = analytics
    
    def register(self, user_data: dict) -> int:
        self._validator.validate(user_data)
        user_id = self._repository.save(user_data)
        self._email.send_welcome(user_data["email"])
        self._analytics.track("user_registered", {"user_id": user_id})
        return user_id
```

---

### 2.2 Open/Closed Principle (OCP)

> **Rule:** Software entities should be open for extension but closed for modification.

**Why it matters:**
- Adding new features shouldn't require changing existing, tested code
- Reduces risk of introducing bugs in working functionality
- Enables plugin-like architectures

**❌ DON'T: Modify existing code for each new case**
```python
class PaymentProcessor:
    def process(self, payment_type: str, amount: float):
        if payment_type == "credit_card":
            print(f"Processing ${amount} via credit card")
        elif payment_type == "paypal":
            print(f"Processing ${amount} via PayPal")
        elif payment_type == "crypto":  # Adding this requires modifying the class!
            print(f"Processing ${amount} via Crypto")
        # This grows forever...
```

**✅ DO: Use abstractions to allow extension**
```python
from abc import ABC, abstractmethod
from typing import Protocol

# Option 1: Abstract Base Class
class PaymentHandler(ABC):
    @abstractmethod
    def process(self, amount: float) -> None:
        """Process the payment."""
        pass
    
    @abstractmethod
    def supports(self, payment_type: str) -> bool:
        """Check if this handler supports the payment type."""
        pass

# Option 2: Protocol (preferred for flexibility)
class PaymentHandler(Protocol):
    def process(self, amount: float) -> None: ...
    def supports(self, payment_type: str) -> bool: ...

# Implementations - each in its own module if needed
class CreditCardHandler:
    def process(self, amount: float) -> None:
        print(f"Processing ${amount} via credit card")
    
    def supports(self, payment_type: str) -> bool:
        return payment_type == "credit_card"

class PayPalHandler:
    def process(self, amount: float) -> None:
        print(f"Processing ${amount} via PayPal")
    
    def supports(self, payment_type: str) -> bool:
        return payment_type == "paypal"

# Adding new payment methods = creating new classes, NO modification needed
class CryptoHandler:
    def process(self, amount: float) -> None:
        print(f"Processing ${amount} via Crypto")
    
    def supports(self, payment_type: str) -> bool:
        return payment_type == "crypto"

# Processor that's CLOSED for modification
class PaymentProcessor:
    def __init__(self, handlers: list[PaymentHandler]):
        self._handlers = handlers
    
    def process(self, payment_type: str, amount: float) -> None:
        for handler in self._handlers:
            if handler.supports(payment_type):
                handler.process(amount)
                return
        raise ValueError(f"No handler for payment type: {payment_type}")

# Usage - extend by adding handlers, not modifying processor
processor = PaymentProcessor([
    CreditCardHandler(),
    PayPalHandler(),
    CryptoHandler(),
])
```

---

### 2.3 Liskov Substitution Principle (LSP)

> **Rule:** Objects of a superclass should be replaceable with objects of a subclass without breaking the application.

**Why it matters:**
- Enables polymorphism to work correctly
- Prevents subtle bugs when using inheritance
- Makes code predictable

**❌ DON'T: Violate the contract of the parent class**
```python
class Bird:
    def fly(self) -> str:
        return "Flying high!"

class Penguin(Bird):
    def fly(self) -> str:
        # Violates LSP - penguins can't fly!
        raise NotImplementedError("Penguins can't fly")

# This breaks when you try to use polymorphism
def make_bird_fly(bird: Bird) -> None:
    print(bird.fly())  # Crashes with Penguin!
```

**✅ DO: Design inheritance hierarchies correctly**
```python
from abc import ABC, abstractmethod

# Better hierarchy based on actual capabilities
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        pass

class FlyingBird(Bird):
    def move(self) -> str:
        return self.fly()
    
    def fly(self) -> str:
        return "Flying high!"

class SwimmingBird(Bird):
    def move(self) -> str:
        return self.swim()
    
    def swim(self) -> str:
        return "Swimming gracefully!"

class Eagle(FlyingBird):
    pass

class Penguin(SwimmingBird):
    pass

# Now polymorphism works correctly
def make_bird_move(bird: Bird) -> None:
    print(bird.move())  # Works for ALL birds!

make_bird_move(Eagle())   # "Flying high!"
make_bird_move(Penguin()) # "Swimming gracefully!"
```

**Key LSP Rules:**
1. **Don't strengthen preconditions:** If parent accepts `int`, child can't require `positive int`
2. **Don't weaken postconditions:** If parent promises to return `list`, child can't return `None`
3. **Don't throw new exceptions:** Child shouldn't raise exceptions parent doesn't raise
4. **Preserve invariants:** If parent guarantees `len(items) > 0`, child must too

---

### 2.4 Interface Segregation Principle (ISP)

> **Rule:** Clients should not be forced to depend on interfaces they don't use.

**Why it matters:**
- Smaller interfaces are easier to implement
- Changes to unused methods won't affect clients
- Promotes composition over inheritance

**❌ DON'T: Create "fat" interfaces**
```python
from abc import ABC, abstractmethod

class Worker(ABC):
    @abstractmethod
    def work(self) -> None: pass
    
    @abstractmethod
    def eat(self) -> None: pass
    
    @abstractmethod
    def sleep(self) -> None: pass
    
    @abstractmethod
    def receive_salary(self) -> None: pass

# Robot can't eat or sleep - forced to implement useless methods!
class Robot(Worker):
    def work(self) -> None:
        print("Working...")
    
    def eat(self) -> None:
        pass  # Meaningless implementation
    
    def sleep(self) -> None:
        pass  # Meaningless implementation
    
    def receive_salary(self) -> None:
        pass  # Robots don't get paid
```

**✅ DO: Use small, focused protocols**
```python
from typing import Protocol

# Segregated interfaces
class Workable(Protocol):
    def work(self) -> None: ...

class Eatable(Protocol):
    def eat(self) -> None: ...

class Sleepable(Protocol):
    def sleep(self) -> None: ...

class Payable(Protocol):
    def receive_salary(self, amount: float) -> None: ...

# Human implements all that apply
class Human:
    def work(self) -> None:
        print("Human working...")
    
    def eat(self) -> None:
        print("Human eating...")
    
    def sleep(self) -> None:
        print("Human sleeping...")
    
    def receive_salary(self, amount: float) -> None:
        print(f"Received ${amount}")

# Robot only implements what it needs
class Robot:
    def work(self) -> None:
        print("Robot working 24/7...")

# Functions depend only on what they need
def manage_work(worker: Workable) -> None:
    worker.work()

def lunch_break(eater: Eatable) -> None:
    eater.eat()

# Both work with manage_work
manage_work(Human())  # ✅
manage_work(Robot())  # ✅

# Only Human works with lunch_break
lunch_break(Human())  # ✅
# lunch_break(Robot())  # ❌ Type error - Robot doesn't have eat()
```

---

### 2.5 Dependency Inversion Principle (DIP)

> **Rule:** High-level modules should not depend on low-level modules. Both should depend on abstractions.

**Why it matters:**
- Decouples components for easier testing
- Allows swapping implementations (e.g., different databases)
- Enables parallel development

**❌ DON'T: Hard-code dependencies**
```python
import sqlite3

class UserService:
    def __init__(self):
        # Tight coupling to SQLite!
        self.db = sqlite3.connect("users.db")
    
    def get_user(self, user_id: int):
        cursor = self.db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        return cursor.fetchone()

# Problems:
# - Can't easily switch to PostgreSQL
# - Can't mock the database for testing
# - UserService knows too much about database implementation
```

**✅ DO: Depend on abstractions, inject dependencies**
```python
from typing import Protocol
from dataclasses import dataclass

# Domain model
@dataclass
class User:
    id: int
    email: str
    name: str

# Abstract interface (what, not how)
class UserRepository(Protocol):
    def get_by_id(self, user_id: int) -> User | None: ...
    def save(self, user: User) -> None: ...

# Concrete implementation 1: SQLite
class SQLiteUserRepository:
    def __init__(self, connection):
        self.conn = connection
    
    def get_by_id(self, user_id: int) -> User | None:
        cursor = self.conn.execute(
            "SELECT id, email, name FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return User(*row) if row else None
    
    def save(self, user: User) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO users VALUES (?, ?, ?)",
            (user.id, user.email, user.name)
        )

# Concrete implementation 2: In-memory (for testing)
class InMemoryUserRepository:
    def __init__(self):
        self._users: dict[int, User] = {}
    
    def get_by_id(self, user_id: int) -> User | None:
        return self._users.get(user_id)
    
    def save(self, user: User) -> None:
        self._users[user.id] = user

# High-level service depends on abstraction
class UserService:
    def __init__(self, repository: UserRepository):  # Injected!
        self._repository = repository
    
    def get_user(self, user_id: int) -> User | None:
        return self._repository.get_by_id(user_id)

# Production
production_service = UserService(SQLiteUserRepository(sqlite_conn))

# Testing
test_service = UserService(InMemoryUserRepository())
```

---

## 3. Additional Design Principles

### 3.1 DRY (Don't Repeat Yourself)

> **Rule:** Every piece of knowledge should have a single, unambiguous representation in the system.

**Why it matters:**
- Single point of change for bug fixes
- Reduces inconsistencies
- Smaller codebase

**❌ DON'T: Copy-paste code**
```python
def create_admin_user(data):
    if not data.get("email"):
        raise ValueError("Email required")
    if "@" not in data["email"]:
        raise ValueError("Invalid email")
    if len(data.get("password", "")) < 8:
        raise ValueError("Password must be 8+ characters")
    # ... create admin

def create_regular_user(data):
    if not data.get("email"):
        raise ValueError("Email required")
    if "@" not in data["email"]:
        raise ValueError("Invalid email")
    if len(data.get("password", "")) < 8:
        raise ValueError("Password must be 8+ characters")
    # ... create regular user
```

**✅ DO: Extract common logic**
```python
# Option 1: Extract to function
def validate_user_data(data: dict) -> None:
    """Validate common user data fields."""
    if not data.get("email"):
        raise ValueError("Email required")
    if "@" not in data["email"]:
        raise ValueError("Invalid email")
    if len(data.get("password", "")) < 8:
        raise ValueError("Password must be 8+ characters")

def create_admin_user(data):
    validate_user_data(data)
    # ... create admin

def create_regular_user(data):
    validate_user_data(data)
    # ... create regular user

# Option 2: Use a decorator for cross-cutting concerns
from functools import wraps

def validate_input(validator_func):
    """Decorator that validates input before calling function."""
    def decorator(func):
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            validator_func(data)
            return func(data, *args, **kwargs)
        return wrapper
    return decorator

@validate_input(validate_user_data)
def create_admin_user(data):
    # ... create admin
    pass

# Option 3: Use Pydantic for declarative validation
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

def create_user(data: UserCreate):  # Validation is automatic
    pass
```

**⚠️ CAUTION:** DRY doesn't mean "merge everything that looks similar." If two pieces of code change for different reasons, they should stay separate even if they look alike today.

---

### 3.2 KISS (Keep It Simple, Stupid)

> **Rule:** The simplest solution that works is usually the best.

**Why it matters:**
- Easier to understand and maintain
- Fewer bugs
- Faster development

**❌ DON'T: Over-engineer simple problems**
```python
# Over-engineered way to check if a number is even
class NumberChecker:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def check(self, number):
        return self.strategy.execute(number)

class EvenCheckStrategy:
    def execute(self, number):
        return number % 2 == 0

checker = NumberChecker(EvenCheckStrategy())
is_even = checker.check(4)
```

**✅ DO: Use the simplest solution**
```python
# Just use a function!
def is_even(number: int) -> bool:
    return number % 2 == 0

# Or inline if used once
if number % 2 == 0:
    print("Even!")
```

**KISS Guidelines:**
| Scenario                      | Simple Solution                            |
| ----------------------------- | ------------------------------------------ |
| Need to store key-value pairs | Use a `dict`, not a custom class           |
| Need to call a function once  | Pass the function, don't create a class    |
| Need to transform a list      | Use comprehension, not `map()` with lambda |
| Need configuration            | Use environment variables or a simple dict |

---

### 3.3 YAGNI (You Ain't Gonna Need It)

> **Rule:** Don't implement something until it's actually needed.

**Why it matters:**
- Unused code is a maintenance burden
- Requirements often change
- Python's dynamic nature makes refactoring easy

**❌ DON'T: Build "just in case" features**
```python
class UserService:
    def get_user(self, user_id: int): ...
    def get_user_async(self, user_id: int): ...  # "Might need async later"
    def get_user_cached(self, user_id: int): ...  # "Might need caching"
    def get_user_with_retry(self, user_id: int): ...  # "Might need retries"
    def get_user_batch(self, user_ids: list): ...  # "Might need batch"
```

**✅ DO: Implement what's needed now**
```python
class UserService:
    def get_user(self, user_id: int) -> User | None:
        """Get a user by ID. That's all we need right now."""
        return self._repository.get_by_id(user_id)

# When (if!) you need caching, add it then:
# from functools import lru_cache
# 
# @lru_cache(maxsize=100)
# def get_user(self, user_id: int) -> User | None:
#     ...
```

---

### 3.4 GRASP Principles

GRASP (General Responsibility Assignment Software Patterns) provides guidance on assigning responsibilities to classes.

#### Information Expert

> **Rule:** Assign responsibility to the class that has the information needed to fulfill it.

**❌ DON'T: Pull data out to process elsewhere**
```python
class Order:
    def __init__(self):
        self.items = []

class OrderProcessor:
    def calculate_total(self, order: Order) -> float:
        # Pulling data out of Order to process here
        total = 0
        for item in order.items:
            total += item.price * item.quantity
        return total
```

**✅ DO: Let the class with the data do the work**
```python
class Order:
    def __init__(self):
        self.items: list[OrderItem] = []
    
    def calculate_total(self) -> float:
        """Order has the items, so it calculates the total."""
        return sum(item.subtotal for item in self.items)

@dataclass
class OrderItem:
    price: float
    quantity: int
    
    @property
    def subtotal(self) -> float:
        """OrderItem knows its price and quantity."""
        return self.price * self.quantity
```

#### Creator

> **Rule:** Assign class B the responsibility to create class A if B contains A, aggregates A, or has the data to initialize A.

```python
class Order:
    def __init__(self):
        self.items: list[OrderItem] = []
    
    def add_item(self, product: Product, quantity: int) -> OrderItem:
        """Order creates OrderItems because it contains them."""
        item = OrderItem(
            product_id=product.id,
            price=product.price,
            quantity=quantity
        )
        self.items.append(item)
        return item
```

#### Controller

> **Rule:** Use a dedicated object to handle system events and coordinate responses.

```python
# API Controller (FastAPI example)
from fastapi import APIRouter, Depends

router = APIRouter()

@router.post("/orders")
async def create_order(
    request: CreateOrderRequest,
    service: OrderService = Depends(get_order_service),  # Injected
):
    """Controller handles HTTP, delegates to service."""
    order = await service.create_order(
        user_id=request.user_id,
        items=request.items,
    )
    return OrderResponse.from_domain(order)

# Business logic stays in the service, unaware of HTTP
class OrderService:
    def create_order(self, user_id: int, items: list[ItemData]) -> Order:
        # Pure business logic, no HTTP knowledge
        pass
```

#### Low Coupling & High Cohesion

> **Rule:** Minimize dependencies between classes (low coupling) while keeping related functionality together (high cohesion).

```python
# HIGH COHESION: All methods relate to user authentication
class AuthenticationService:
    def login(self, credentials: Credentials) -> Token: ...
    def logout(self, token: Token) -> None: ...
    def refresh_token(self, token: Token) -> Token: ...
    def validate_token(self, token: Token) -> bool: ...

# LOW COUPLING: Services communicate through abstractions
class OrderService:
    def __init__(
        self,
        auth_service: AuthService,  # Abstract interface
        payment_service: PaymentService,  # Abstract interface
    ):
        self._auth = auth_service
        self._payment = payment_service
```

---

## 4. GoF Design Patterns in Python

The Gang of Four patterns apply to Python but often have simpler, more Pythonic implementations.

### 4.1 Creational Patterns

#### Factory Pattern

**When to use:** Creating objects without specifying exact classes.

```python
from typing import Protocol
from enum import Enum, auto

class NotificationChannel(Protocol):
    def send(self, message: str) -> None: ...

class EmailNotification:
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class SMSNotification:
    def send(self, message: str) -> None:
        print(f"SMS: {message}")

class PushNotification:
    def send(self, message: str) -> None:
        print(f"Push: {message}")

class ChannelType(Enum):
    EMAIL = auto()
    SMS = auto()
    PUSH = auto()

# Simple factory function (Pythonic approach)
def create_notification(channel: ChannelType) -> NotificationChannel:
    """Factory function - simpler than a factory class in Python."""
    factories = {
        ChannelType.EMAIL: EmailNotification,
        ChannelType.SMS: SMSNotification,
        ChannelType.PUSH: PushNotification,
    }
    
    if channel not in factories:
        raise ValueError(f"Unknown channel: {channel}")
    
    return factories[channel]()

# Usage
notification = create_notification(ChannelType.EMAIL)
notification.send("Hello!")
```

#### Singleton Pattern

**When to use:** Ensuring only one instance exists (use sparingly!).

```python
# ❌ DON'T: Classic singleton (over-engineered for Python)
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# ✅ DO: Use module-level instance (Pythonic singleton)
# config.py
class _Config:
    def __init__(self):
        self.debug = False
        self.database_url = ""

# Module-level instance acts as singleton
config = _Config()

# Other files just import it
# from config import config
# config.debug = True

# ✅ BETTER: Use functools.lru_cache for expensive initialization
from functools import lru_cache

@lru_cache(maxsize=1)
def get_database_connection():
    """Called once, result cached forever."""
    return create_expensive_connection()
```

#### Builder Pattern

**When to use:** Constructing complex objects step by step.

```python
from dataclasses import dataclass, field
from typing import Self

@dataclass
class Email:
    recipient: str
    subject: str
    body: str
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

class EmailBuilder:
    """Builder for complex Email objects."""
    
    def __init__(self):
        self._recipient: str = ""
        self._subject: str = ""
        self._body: str = ""
        self._cc: list[str] = []
        self._bcc: list[str] = []
        self._attachments: list[str] = []
    
    def to(self, recipient: str) -> Self:  # Python 3.11+ Self type
        self._recipient = recipient
        return self
    
    def with_subject(self, subject: str) -> Self:
        self._subject = subject
        return self
    
    def with_body(self, body: str) -> Self:
        self._body = body
        return self
    
    def add_cc(self, *addresses: str) -> Self:
        self._cc.extend(addresses)
        return self
    
    def add_attachment(self, path: str) -> Self:
        self._attachments.append(path)
        return self
    
    def build(self) -> Email:
        if not self._recipient:
            raise ValueError("Recipient is required")
        if not self._subject:
            raise ValueError("Subject is required")
        
        return Email(
            recipient=self._recipient,
            subject=self._subject,
            body=self._body,
            cc=self._cc,
            bcc=self._bcc,
            attachments=self._attachments,
        )

# Usage with fluent interface
email = (
    EmailBuilder()
    .to("user@example.com")
    .with_subject("Hello")
    .with_body("This is the body")
    .add_cc("manager@example.com", "team@example.com")
    .add_attachment("/path/to/file.pdf")
    .build()
)
```

### 4.2 Structural Patterns

#### Adapter Pattern

**When to use:** Making incompatible interfaces work together.

```python
from typing import Protocol

# Your application expects this interface
class PaymentGateway(Protocol):
    def charge(self, amount: float, currency: str) -> str: ...

# Third-party library has a different interface
class LegacyPaymentSDK:
    def make_payment(self, cents: int, currency_code: str) -> dict:
        return {"transaction_id": "txn_123", "status": "success"}

# Adapter makes legacy SDK work with your interface
class LegacyPaymentAdapter:
    """Adapts LegacyPaymentSDK to PaymentGateway interface."""
    
    def __init__(self, legacy_sdk: LegacyPaymentSDK):
        self._sdk = legacy_sdk
    
    def charge(self, amount: float, currency: str) -> str:
        # Convert dollars to cents, adapt the call
        cents = int(amount * 100)
        result = self._sdk.make_payment(cents, currency.upper())
        return result["transaction_id"]

# Usage - your code works with the abstract interface
def process_order(gateway: PaymentGateway, amount: float):
    transaction_id = gateway.charge(amount, "usd")
    print(f"Charged! Transaction: {transaction_id}")

# Inject the adapter
legacy_sdk = LegacyPaymentSDK()
adapter = LegacyPaymentAdapter(legacy_sdk)
process_order(adapter, 99.99)
```

#### Decorator Pattern

**When to use:** Adding behavior dynamically without inheritance.

```python
from typing import Protocol
from functools import wraps
import time
import logging

# Python's built-in decorator syntax is the idiomatic approach

# Option 1: Function decorator (most common)
def log_calls(func):
    """Decorator that logs function calls."""
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper

def measure_time(func):
    """Decorator that measures execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator factory for retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# Decorators can be stacked
@log_calls
@measure_time
@retry(max_attempts=3)
def fetch_data(url: str) -> dict:
    # Implementation
    pass

# Option 2: Class-based decorator for stateful decoration
class CacheDecorator:
    """Decorator class with state (caching)."""
    
    def __init__(self, func):
        self._func = func
        self._cache = {}
        wraps(func)(self)
    
    def __call__(self, *args):
        if args not in self._cache:
            self._cache[args] = self._func(*args)
        return self._cache[args]
    
    def clear_cache(self):
        self._cache.clear()

@CacheDecorator
def expensive_computation(n: int) -> int:
    return sum(i ** 2 for i in range(n))

expensive_computation.clear_cache()  # Access decorator method
```

### 4.3 Behavioral Patterns

#### Strategy Pattern

**When to use:** Selecting algorithms at runtime.

```python
from typing import Protocol, Callable
from dataclasses import dataclass

# Option 1: Protocol-based (traditional OOP approach)
class CompressionStrategy(Protocol):
    def compress(self, data: bytes) -> bytes: ...

class ZipCompression:
    def compress(self, data: bytes) -> bytes:
        import zlib
        return zlib.compress(data)

class GzipCompression:
    def compress(self, data: bytes) -> bytes:
        import gzip
        return gzip.compress(data)

class Compressor:
    def __init__(self, strategy: CompressionStrategy):
        self._strategy = strategy
    
    def compress_file(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return self._strategy.compress(f.read())

# Option 2: Function-based (more Pythonic for simple strategies)
CompressionFunc = Callable[[bytes], bytes]

def zip_compress(data: bytes) -> bytes:
    import zlib
    return zlib.compress(data)

def gzip_compress(data: bytes) -> bytes:
    import gzip
    return gzip.compress(data)

def compress_file(path: str, strategy: CompressionFunc) -> bytes:
    with open(path, "rb") as f:
        return strategy(f.read())

# Usage
result = compress_file("data.txt", zip_compress)
result = compress_file("data.txt", gzip_compress)
```

#### Observer Pattern

**When to use:** Notifying multiple objects of state changes.

```python
from typing import Protocol, Callable
from dataclasses import dataclass, field

# Option 1: Protocol-based
class Observer(Protocol):
    def update(self, event: str, data: dict) -> None: ...

class EventEmitter:
    def __init__(self):
        self._observers: list[Observer] = []
    
    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def unsubscribe(self, observer: Observer) -> None:
        self._observers.remove(observer)
    
    def emit(self, event: str, data: dict) -> None:
        for observer in self._observers:
            observer.update(event, data)

# Option 2: Callback-based (more Pythonic)
EventHandler = Callable[[dict], None]

@dataclass
class EventBus:
    _handlers: dict[str, list[EventHandler]] = field(default_factory=dict)
    
    def subscribe(self, event: str, handler: EventHandler) -> None:
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        if event in self._handlers:
            self._handlers[event].remove(handler)
    
    def emit(self, event: str, data: dict) -> None:
        for handler in self._handlers.get(event, []):
            handler(data)

# Usage
bus = EventBus()

def on_user_created(data: dict):
    print(f"User created: {data['email']}")

def send_welcome_email(data: dict):
    print(f"Sending welcome email to {data['email']}")

bus.subscribe("user_created", on_user_created)
bus.subscribe("user_created", send_welcome_email)

bus.emit("user_created", {"email": "user@example.com"})
```

#### Command Pattern

**When to use:** Encapsulating actions as objects for undo/redo, queuing, or logging.

```python
from typing import Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

class Command(Protocol):
    def execute(self) -> None: ...
    def undo(self) -> None: ...

@dataclass
class AddTextCommand:
    """Command to add text to a document."""
    document: "Document"
    text: str
    position: int = 0
    
    def execute(self) -> None:
        self.document.insert(self.position, self.text)
    
    def undo(self) -> None:
        self.document.delete(self.position, len(self.text))

@dataclass
class DeleteTextCommand:
    """Command to delete text from a document."""
    document: "Document"
    position: int
    length: int
    _deleted_text: str = ""
    
    def execute(self) -> None:
        self._deleted_text = self.document.get_text(self.position, self.length)
        self.document.delete(self.position, self.length)
    
    def undo(self) -> None:
        self.document.insert(self.position, self._deleted_text)

class CommandHistory:
    """Manages command execution with undo/redo support."""
    
    def __init__(self):
        self._history: list[Command] = []
        self._redo_stack: list[Command] = []
    
    def execute(self, command: Command) -> None:
        command.execute()
        self._history.append(command)
        self._redo_stack.clear()  # Clear redo after new command
    
    def undo(self) -> None:
        if self._history:
            command = self._history.pop()
            command.undo()
            self._redo_stack.append(command)
    
    def redo(self) -> None:
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.execute()
            self._history.append(command)
```

---

## 5. Clean Code Practices

### 5.1 Naming Conventions

#### Variables

| Type                    | Convention                      | Examples                                         |
| ----------------------- | ------------------------------- | ------------------------------------------------ |
| **Local variables**     | `snake_case`, descriptive nouns | `user_count`, `total_price`, `active_sessions`   |
| **Constants**           | `UPPER_SNAKE_CASE`              | `MAX_RETRIES`, `DEFAULT_TIMEOUT`, `API_BASE_URL` |
| **Private attributes**  | `_single_underscore` prefix     | `self._cache`, `self._connection`                |
| **"Dunder" names**      | Reserved for Python             | `__init__`, `__str__`, `__dict__`                |
| **Throwaway variables** | `_` single underscore           | `for _ in range(10):`                            |

**❌ DON'T: Use meaningless or misleading names**
```python
# Bad names
d = get_data()  # What kind of data?
temp = process(x)  # Temp what?
flag = True  # Flag for what?
list1 = []  # List of what?
data2 = transform(data1)  # What transformation?
```

**✅ DO: Use intention-revealing names**
```python
# Good names
user_profile = get_user_profile(user_id)
sanitized_input = sanitize_html(raw_input)
is_authenticated = True
pending_orders = []
normalized_scores = normalize_to_percentage(raw_scores)
```

#### Functions and Methods

| Pattern           | When to Use               | Examples                                             |
| ----------------- | ------------------------- | ---------------------------------------------------- |
| **verb_noun**     | Actions that do something | `create_user()`, `send_email()`, `calculate_total()` |
| **get_noun**      | Retrieving data           | `get_user_by_id()`, `get_settings()`                 |
| **is_/has_/can_** | Boolean returns           | `is_valid()`, `has_permission()`, `can_edit()`       |
| **to_format**     | Conversions               | `to_json()`, `to_dict()`, `to_datetime()`            |
| **noun**          | Properties (no verb)      | `@property def total(self):`                         |

**❌ DON'T: Use vague function names**
```python
def process(data): ...  # Process how?
def handle(request): ...  # Handle what?
def do_stuff(): ...  # What stuff?
def manage_users(action, user): ...  # Too many responsibilities
```

**✅ DO: Use specific, descriptive names**
```python
def validate_email_format(email: str) -> bool: ...
def parse_csv_to_dataframe(file_path: str) -> DataFrame: ...
def send_password_reset_email(user: User) -> None: ...
def archive_inactive_users(days_inactive: int = 90) -> int: ...
```

#### Classes

| Convention                              | Examples                                                  |
| --------------------------------------- | --------------------------------------------------------- |
| `PascalCase` nouns                      | `UserRepository`, `PaymentService`, `OrderItem`           |
| Avoid "Manager", "Handler", "Processor" | These often indicate SRP violations                       |
| Suffixes indicate role                  | `Service`, `Repository`, `Factory`, `Strategy`, `Builder` |

### 5.2 Function Design

#### Size Guidelines

> **Rule:** A function should do one thing, do it well, and do it only.

| Metric                    | Guideline                                 |
| ------------------------- | ----------------------------------------- |
| **Lines of code**         | Ideally < 20 lines, max 50 lines          |
| **Arguments**             | Max 3-4; use dataclass/TypedDict for more |
| **Indentation depth**     | Max 2-3 levels; use early returns         |
| **Cyclomatic complexity** | Max 10; break up complex logic            |

**❌ DON'T: Write long functions with deep nesting**
```python
def process_order(order_data, user, settings, db, mailer, logger):
    if order_data:
        if user:
            if user.is_active:
                if settings.orders_enabled:
                    try:
                        items = []
                        for item_data in order_data['items']:
                            if item_data.get('product_id'):
                                product = db.get_product(item_data['product_id'])
                                if product:
                                    if product.in_stock:
                                        if item_data.get('quantity', 0) > 0:
                                            # ... 50 more lines of nested logic
                                            pass
```

**✅ DO: Use early returns and extraction**
```python
@dataclass
class OrderRequest:
    """Group related parameters into a dataclass."""
    user_id: int
    items: list[OrderItemRequest]
    shipping_address: str

def process_order(request: OrderRequest) -> Order:
    """Process a customer order."""
    user = _get_verified_user(request.user_id)
    validated_items = _validate_items(request.items)
    order = _create_order(user, validated_items, request.shipping_address)
    _send_confirmation(user, order)
    return order

def _get_verified_user(user_id: int) -> User:
    """Get user and verify they can place orders."""
    user = user_repository.get_by_id(user_id)
    if user is None:
        raise UserNotFoundError(user_id)
    if not user.is_active:
        raise UserInactiveError(user_id)
    return user

def _validate_items(items: list[OrderItemRequest]) -> list[ValidatedItem]:
    """Validate all order items are available."""
    validated = []
    for item in items:
        product = _get_available_product(item.product_id, item.quantity)
        validated.append(ValidatedItem(product=product, quantity=item.quantity))
    return validated

def _get_available_product(product_id: int, quantity: int) -> Product:
    """Get product and verify availability."""
    product = product_repository.get_by_id(product_id)
    if product is None:
        raise ProductNotFoundError(product_id)
    if product.stock < quantity:
        raise InsufficientStockError(product_id, product.stock, quantity)
    return product
```

#### Guard Clauses

> **Rule:** Handle edge cases early to keep the main logic at the top level.

**❌ DON'T: Use deep nesting for validation**
```python
def calculate_discount(user, order):
    if user is not None:
        if order is not None:
            if len(order.items) > 0:
                if user.membership_level == "gold":
                    return order.total * 0.2
                else:
                    return order.total * 0.1
            else:
                return 0
        else:
            raise ValueError("Order required")
    else:
        raise ValueError("User required")
```

**✅ DO: Use guard clauses for early exit**
```python
def calculate_discount(user: User | None, order: Order | None) -> float:
    """Calculate discount based on user membership."""
    # Guard clauses - handle edge cases first
    if user is None:
        raise ValueError("User required")
    if order is None:
        raise ValueError("Order required")
    if not order.items:
        return 0.0
    
    # Main logic - flat and clear
    discount_rate = 0.2 if user.membership_level == "gold" else 0.1
    return order.total * discount_rate
```

#### Command-Query Separation (CQS)

> **Rule:** Functions should either change state (command) OR return data (query), not both.

**❌ DON'T: Mix commands and queries**
```python
class ShoppingCart:
    def add_and_get_total(self, item: Item) -> float:
        """Bad: Modifies state AND returns a value."""
        self.items.append(item)  # Command (side effect)
        return sum(i.price for i in self.items)  # Query (return value)
```

**✅ DO: Separate commands from queries**
```python
class ShoppingCart:
    def add_item(self, item: Item) -> None:
        """Command: Modifies state, returns nothing."""
        self.items.append(item)
    
    @property
    def total(self) -> float:
        """Query: Returns data, no side effects."""
        return sum(item.price for item in self.items)

# Usage
cart.add_item(item)
print(cart.total)
```

### 5.3 Class Design

#### Prefer Composition Over Inheritance

> **Rule:** Use inheritance for "is-a" relationships; use composition for "has-a" relationships.

**❌ DON'T: Overuse inheritance**
```python
class DatabaseConnection:
    def connect(self): ...
    def execute(self, query): ...

# Bad: UserService is NOT a database connection
class UserService(DatabaseConnection):
    def get_user(self, user_id):
        return self.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

**✅ DO: Use composition**
```python
class UserService:
    def __init__(self, db: DatabaseConnection):
        self._db = db  # Has-a relationship
    
    def get_user(self, user_id: int) -> User:
        result = self._db.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        return User(**result)
```

#### Data Classes for Value Objects

```python
from dataclasses import dataclass, field
from typing import Self

@dataclass(frozen=True)  # Immutable value object
class Money:
    amount: int  # Store as cents to avoid float issues
    currency: str = "USD"
    
    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount / 100:.2f}"

@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str
    
    def format(self) -> str:
        return f"{self.street}, {self.city}, {self.postal_code}, {self.country}"

@dataclass
class User:
    id: int
    email: str
    name: str
    addresses: list[Address] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
```

### 5.4 Documentation

#### Docstrings (Google Style)

```python
def fetch_user_orders(
    user_id: int,
    status: OrderStatus | None = None,
    limit: int = 100,
) -> list[Order]:
    """Fetch orders for a specific user with optional filtering.
    
    Retrieves orders from the database for the given user. Orders are
    returned in descending chronological order (newest first).
    
    Args:
        user_id: The unique identifier of the user.
        status: Optional filter for order status. If None, returns all statuses.
        limit: Maximum number of orders to return. Defaults to 100.
    
    Returns:
        A list of Order objects. Empty list if user has no orders.
    
    Raises:
        UserNotFoundError: If no user exists with the given ID.
        DatabaseError: If there's a problem connecting to the database.
    
    Example:
        >>> orders = fetch_user_orders(123, status=OrderStatus.PENDING)
        >>> print(f"Found {len(orders)} pending orders")
    """
    ...
```

#### When to Comment

| Do Comment                                  | Don't Comment                                            |
| ------------------------------------------- | -------------------------------------------------------- |
| **Why** something is done (business reason) | **What** the code does (code should be self-documenting) |
| Complex algorithms or regex                 | Obvious operations                                       |
| Workarounds for bugs (link to issue)        | Restating the code in English                            |
| Public API usage examples                   | Every function                                           |

```python
# ✅ Good comment - explains WHY
# Using binary search instead of linear search because the user list
# can contain up to 10 million entries (see performance analysis in JIRA-1234)
user_index = bisect.bisect_left(sorted_users, target_user)

# ✅ Good comment - explains business rule
# Tax is calculated on subtotal BEFORE discounts per IRS regulation 1234-A
tax = calculate_tax(subtotal)

# ❌ Bad comment - restates the obvious
# Increment counter by 1
counter += 1

# ❌ Bad comment - what, not why
# Loop through users
for user in users:
    ...
```

---

## 6. Modern Python Features (3.10–3.13)

### 6.1 Structural Pattern Matching (3.10+)

**What it replaces:** Complex `if/elif/else` chains for parsing data structures.

**When to use:**
- Parsing JSON/dict structures with varying shapes
- Handling multiple message types
- Implementing state machines

```python
from dataclasses import dataclass
from typing import Any

# Example 1: Matching dictionary structures
def handle_api_response(response: dict[str, Any]) -> str:
    match response:
        case {"status": "success", "data": {"user": {"id": user_id, "name": name}}}:
            return f"User {name} (ID: {user_id}) loaded successfully"
        
        case {"status": "error", "error": {"code": code, "message": msg}}:
            return f"Error {code}: {msg}"
        
        case {"status": "pending", "retry_after": int(seconds)}:
            return f"Please retry after {seconds} seconds"
        
        case _:
            return "Unknown response format"

# Example 2: Matching with guards
def categorize_number(value: int | float) -> str:
    match value:
        case int(n) if n < 0:
            return "negative integer"
        case int(n) if n == 0:
            return "zero"
        case int(n) if n > 0:
            return "positive integer"
        case float(f) if f.is_integer():
            return "float (whole number)"
        case float():
            return "float (decimal)"
        case _:
            return "not a number"

# Example 3: Matching class instances
@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

@dataclass
class Rectangle:
    top_left: Point
    width: float
    height: float

def calculate_area(shape) -> float:
    match shape:
        case Circle(radius=r):
            return 3.14159 * r ** 2
        case Rectangle(width=w, height=h):
            return w * h
        case Point():
            return 0.0
        case _:
            raise TypeError(f"Unknown shape: {type(shape)}")

# Example 4: Matching sequences
def process_command(command: list[str]) -> None:
    match command:
        case ["quit" | "exit"]:
            print("Goodbye!")
        case ["load", filename]:
            print(f"Loading {filename}")
        case ["save", filename, *options]:
            print(f"Saving {filename} with options: {options}")
        case ["move", x, y] if x.isdigit() and y.isdigit():
            print(f"Moving to ({x}, {y})")
        case [cmd, *_]:
            print(f"Unknown command: {cmd}")
        case []:
            print("No command provided")
```

### 6.2 Exception Groups and except* (3.11+)

**What it replaces:** Manual exception aggregation in concurrent code.

**When to use:** When multiple operations can fail independently and you want to handle all errors.

```python
import asyncio

# Example 1: Handling multiple exceptions from concurrent tasks
async def fetch_all_data():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_users())
        task2 = tg.create_task(fetch_orders())
        task3 = tg.create_task(fetch_products())
    # If any task fails, TaskGroup raises ExceptionGroup

async def main():
    try:
        await fetch_all_data()
    except* ValueError as eg:
        # Handle all ValueErrors
        for exc in eg.exceptions:
            print(f"Validation error: {exc}")
    except* ConnectionError as eg:
        # Handle all ConnectionErrors
        for exc in eg.exceptions:
            print(f"Connection error: {exc}")

# Example 2: Creating exception groups manually
def validate_user_data(data: dict) -> None:
    errors = []
    
    if not data.get("email"):
        errors.append(ValueError("Email is required"))
    elif "@" not in data["email"]:
        errors.append(ValueError("Invalid email format"))
    
    if not data.get("password"):
        errors.append(ValueError("Password is required"))
    elif len(data["password"]) < 8:
        errors.append(ValueError("Password must be at least 8 characters"))
    
    if not data.get("age"):
        errors.append(ValueError("Age is required"))
    elif data["age"] < 18:
        errors.append(ValueError("Must be 18 or older"))
    
    if errors:
        raise ExceptionGroup("Validation failed", errors)

# Handling
try:
    validate_user_data({"email": "bad", "password": "123", "age": 15})
except* ValueError as eg:
    for error in eg.exceptions:
        print(f"- {error}")
```

### 6.3 TaskGroup for Structured Concurrency (3.11+)

**What it replaces:** `asyncio.gather()` with inconsistent error handling.

**Why it's better:**
- Automatic cancellation of remaining tasks when one fails
- Proper error propagation via `ExceptionGroup`
- Cannot accidentally "forget" to await tasks

```python
import asyncio
from dataclasses import dataclass

@dataclass
class UserDashboard:
    user: dict
    orders: list
    recommendations: list

# ❌ DON'T: Use asyncio.gather() without proper error handling
async def get_dashboard_old(user_id: int) -> UserDashboard:
    # If fetch_orders fails, fetch_recommendations keeps running wastefully
    # Errors are returned, not raised (return_exceptions=True) or
    # first error raised, others lost (return_exceptions=False)
    user, orders, recs = await asyncio.gather(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_recommendations(user_id),
    )
    return UserDashboard(user, orders, recs)

# ✅ DO: Use TaskGroup for proper structured concurrency
async def get_dashboard(user_id: int) -> UserDashboard:
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user(user_id))
        orders_task = tg.create_task(fetch_orders(user_id))
        recs_task = tg.create_task(fetch_recommendations(user_id))
    
    # All tasks completed successfully (or ExceptionGroup was raised)
    return UserDashboard(
        user=user_task.result(),
        orders=orders_task.result(),
        recommendations=recs_task.result(),
    )

# Example with timeout and cancellation
async def get_dashboard_with_timeout(user_id: int) -> UserDashboard:
    try:
        async with asyncio.timeout(5.0):  # 3.11+ timeout context
            async with asyncio.TaskGroup() as tg:
                user_task = tg.create_task(fetch_user(user_id))
                orders_task = tg.create_task(fetch_orders(user_id))
                recs_task = tg.create_task(fetch_recommendations(user_id))
        
        return UserDashboard(
            user=user_task.result(),
            orders=orders_task.result(),
            recommendations=recs_task.result(),
        )
    except TimeoutError:
        raise DashboardTimeoutError(f"Dashboard for user {user_id} timed out")
    except* DatabaseError as eg:
        raise DashboardDatabaseError(f"Database errors: {eg.exceptions}")
```

### 6.4 Improved Typing Features

#### `Self` Type (3.11+)

**What it replaces:** `TypeVar` bound to the class for fluent interfaces.

```python
from typing import Self

# ❌ OLD WAY (pre-3.11)
from typing import TypeVar
T = TypeVar("T", bound="Builder")

class BuilderOld:
    def set_name(self: T, name: str) -> T:
        self.name = name
        return self

# ✅ NEW WAY (3.11+)
class Builder:
    def __init__(self):
        self.name = ""
        self.value = 0
    
    def set_name(self, name: str) -> Self:
        self.name = name
        return self
    
    def set_value(self, value: int) -> Self:
        self.value = value
        return self
    
    def clone(self) -> Self:
        """Returns a copy of the same type."""
        new = type(self)()
        new.name = self.name
        new.value = self.value
        return new

# Works correctly with inheritance
class AdvancedBuilder(Builder):
    def set_extra(self, extra: str) -> Self:  # Returns AdvancedBuilder
        self.extra = extra
        return self

result = AdvancedBuilder().set_name("test").set_extra("more")  # Type: AdvancedBuilder
```

#### `LiteralString` (3.11+)

**When to use:** Preventing SQL injection and similar vulnerabilities.

```python
from typing import LiteralString

def execute_query(query: LiteralString) -> list[dict]:
    """Only accepts string literals, not dynamic strings."""
    # Safe to execute because query must be a literal
    return database.execute(query)

# ✅ These work
execute_query("SELECT * FROM users")
execute_query("SELECT * FROM users WHERE id = ?")

# ❌ These cause type errors
user_input = input("Enter query: ")
execute_query(user_input)  # Type error!

table = "users"
execute_query(f"SELECT * FROM {table}")  # Type error! (f-string is not literal)
```

#### `Required` and `NotRequired` for TypedDict (3.11+)

```python
from typing import TypedDict, Required, NotRequired

# Pre-3.11: Had to create two TypedDicts and inherit
# class UserBase(TypedDict):
#     id: int
#     name: str
# class User(UserBase, total=False):
#     email: str

# 3.11+: Use Required/NotRequired in one definition
class User(TypedDict):
    id: Required[int]  # Must be present
    name: Required[str]  # Must be present
    email: NotRequired[str]  # Optional
    phone: NotRequired[str]  # Optional

def create_user(data: User) -> None:
    user_id = data["id"]  # Safe, always present
    name = data["name"]  # Safe, always present
    email = data.get("email")  # Might be None
```

#### `override` Decorator (3.12+)

**What it replaces:** Manual verification that you're actually overriding something.

```python
from typing import override

class Parent:
    def process(self, data: str) -> int:
        return len(data)

class Child(Parent):
    @override  # Tells type checker this MUST override a parent method
    def process(self, data: str) -> int:
        return super().process(data) * 2
    
    @override  # Type error! No such method in parent
    def proccess(self, data: str) -> int:  # Typo won't go unnoticed
        return 0
```

#### Type Parameter Syntax (3.12+)

**What it replaces:** Explicit `TypeVar` declarations.

```python
# ❌ OLD WAY (pre-3.12)
from typing import TypeVar, Generic

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

class ContainerOld(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value

def first_old(items: list[T]) -> T | None:
    return items[0] if items else None

# ✅ NEW WAY (3.12+) - inline type parameters
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value

def first[T](items: list[T]) -> T | None:
    return items[0] if items else None

def transform[T, U](value: T, func: Callable[[T], U]) -> U:
    return func(value)

# Type aliases with type parameters (3.12+)
type Vector[T] = list[T]
type Matrix[T] = list[Vector[T]]
type Pair[K, V] = tuple[K, V]

# Usage
numbers: Vector[int] = [1, 2, 3]
grid: Matrix[float] = [[1.0, 2.0], [3.0, 4.0]]
```

### 6.5 Union Syntax with `|` (3.10+)

**What it replaces:** `Union` and `Optional` from `typing` module.

```python
# ❌ OLD WAY
from typing import Union, Optional, List, Dict

def process(value: Union[str, int]) -> Optional[str]:
    ...

def get_items() -> List[Dict[str, Union[str, int]]]:
    ...

# ✅ NEW WAY (3.10+)
def process(value: str | int) -> str | None:
    ...

def get_items() -> list[dict[str, str | int]]:
    ...

# Also works in isinstance() checks (3.10+)
if isinstance(value, str | int):
    print("It's a string or int")
```

### 6.6 Walrus Operator `:=` (3.8+)

**What it replaces:** Separate assignment and conditional statements.

```python
# Example 1: While loops with assignment
# ❌ OLD WAY
while True:
    line = file.readline()
    if not line:
        break
    process(line)

# ✅ NEW WAY
while (line := file.readline()):
    process(line)

# Example 2: Conditional with expensive computation
# ❌ OLD WAY
match = pattern.search(text)
if match:
    process(match.group())

# ✅ NEW WAY
if (match := pattern.search(text)):
    process(match.group())

# Example 3: List comprehension with filtering
# ❌ OLD WAY - calls expensive() twice
results = [expensive(x) for x in items if expensive(x) > threshold]

# ✅ NEW WAY - calls expensive() once
results = [y for x in items if (y := expensive(x)) > threshold]

# Example 4: Any/all with capture
# ❌ OLD WAY
found = None
for item in items:
    if is_valid(item):
        found = item
        break

# ✅ NEW WAY (combined with next())
found = next((item for item in items if is_valid(item)), None)

# Or with any() when you need the value
if any((valid := item) for item in items if is_valid(item)):
    print(f"Found: {valid}")
```

### 6.7 Positional-Only and Keyword-Only Arguments (3.8+)

**When to use:**
- Positional-only (`/`): When parameter names are implementation details
- Keyword-only (`*`): When parameters should always be explicit

```python
def create_user(
    # Positional-only: callers can't use 'name=' or 'email='
    name,
    email,
    /,
    # Either positional or keyword
    role="user",
    *,
    # Keyword-only: callers MUST use 'is_active=', 'notify='
    is_active: bool = True,
    notify: bool = False,
) -> User:
    ...

# Valid calls
create_user("John", "john@example.com")
create_user("John", "john@example.com", "admin")
create_user("John", "john@example.com", role="admin")
create_user("John", "john@example.com", is_active=False)
create_user("John", "john@example.com", "admin", is_active=True, notify=True)

# Invalid calls
create_user(name="John", email="john@example.com")  # Error: positional-only
create_user("John", "john@example.com", True)  # Error: is_active is keyword-only
```

---

## 7. Type System & Generics

### 7.1 Type Hints Best Practices

#### When to Use Type Hints

| Always Type                           | Optional                           |
| ------------------------------------- | ---------------------------------- |
| Function signatures (params + return) | Local variables (usually inferred) |
| Class attributes                      | Lambda expressions                 |
| Public APIs                           | Very short scripts                 |
| Module-level variables                |                                    |

#### Common Type Patterns

```python
from typing import (
    Any, 
    Callable, 
    ClassVar, 
    Final, 
    Literal, 
    TypeAlias,
)
from collections.abc import Sequence, Mapping, Iterable, Iterator

# Basic types
name: str = "John"
age: int = 30
price: float = 19.99
active: bool = True

# Collections (use built-in generics, not typing.List, etc.)
names: list[str] = ["Alice", "Bob"]
ages: dict[str, int] = {"Alice": 30, "Bob": 25}
unique_ids: set[int] = {1, 2, 3}
coordinates: tuple[float, float] = (1.0, 2.0)
mixed_tuple: tuple[str, int, bool] = ("test", 1, True)
any_length_tuple: tuple[int, ...] = (1, 2, 3, 4, 5)

# Optional (can be None)
maybe_name: str | None = None

# Union (one of multiple types)
id_value: int | str = "abc123"

# Literal (specific values only)
status: Literal["pending", "approved", "rejected"] = "pending"

# Final (constant, can't be reassigned)
MAX_RETRIES: Final = 3

# ClassVar (class-level, not instance-level)
class MyClass:
    instance_count: ClassVar[int] = 0

# Type aliases (for readability)
UserId: TypeAlias = int
UserDict: TypeAlias = dict[str, str | int | None]
Handler: TypeAlias = Callable[[str], None]

# Callable types
def process(handler: Callable[[int, str], bool]) -> None:
    result = handler(1, "test")

# With *args and **kwargs
def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

# Abstract collection types (accept more input types)
def process_items(items: Iterable[int]) -> list[int]:
    """Accept any iterable (list, tuple, set, generator, etc.)."""
    return [x * 2 for x in items]

def lookup(data: Mapping[str, int], key: str) -> int | None:
    """Accept any mapping (dict, OrderedDict, etc.)."""
    return data.get(key)

def get_head(items: Sequence[str]) -> str | None:
    """Accept any sequence (list, tuple, str, etc.)."""
    return items[0] if items else None
```

### 7.2 Generics

```python
from typing import TypeVar, Generic, Protocol
from collections.abc import Callable

# Simple generic function (3.12+ syntax)
def identity[T](value: T) -> T:
    return value

# Pre-3.12 syntax (still valid)
T = TypeVar("T")

def identity_old(value: T) -> T:
    return value

# Bounded generics (T must be a subtype)
from numbers import Number

def add_numbers[T: Number](a: T, b: T) -> T:
    return a + b  # type: ignore

# Pre-3.12 bounded generic
NumT = TypeVar("NumT", bound=Number)

# Constrained generics (T must be one of specific types)
def format_value[T: (int, float, str)](value: T) -> str:
    return str(value)

# Generic classes
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self) -> T | None:
        return self._items[-1] if self._items else None
    
    def __len__(self) -> int:
        return len(self._items)

# Usage
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")

# Multiple type parameters
class Pair[K, V]:
    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
    
    def swap(self) -> "Pair[V, K]":
        return Pair(self.value, self.key)

# Generic protocol
class Comparable[T](Protocol):
    def __lt__(self, other: T) -> bool: ...
    def __gt__(self, other: T) -> bool: ...

def find_max[T: Comparable[T]](items: list[T]) -> T:
    if not items:
        raise ValueError("Empty list")
    return max(items)
```

### 7.3 Protocol vs ABC

| Feature                      | `Protocol`                          | `ABC`                          |
| ---------------------------- | ----------------------------------- | ------------------------------ |
| Subclassing required         | No (structural typing)              | Yes (nominal typing)           |
| `isinstance()` check         | Only with `@runtime_checkable`      | Yes                            |
| Mixing with concrete methods | No                                  | Yes                            |
| Best for                     | Defining interfaces for duck typing | Providing base implementations |

```python
from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod

# Protocol: Structural typing (duck typing with type safety)
@runtime_checkable  # Enables isinstance() checks
class Drawable(Protocol):
    def draw(self) -> None: ...

# No inheritance needed! Any class with draw() matches
class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

render(Circle())  # ✅ Works
render(Square())  # ✅ Works

# ABC: Nominal typing with shared implementation
class Shape(ABC):
    def __init__(self, color: str):
        self.color = color  # Shared state
    
    @abstractmethod
    def area(self) -> float:
        """Must be implemented by subclasses."""
        pass
    
    def describe(self) -> str:
        """Shared implementation."""
        return f"A {self.color} shape with area {self.area()}"

class Rectangle(Shape):
    def __init__(self, color: str, width: float, height: float):
        super().__init__(color)
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height

# Use Protocol when:
# - You want to accept any object with certain methods (duck typing)
# - You don't control the classes (third-party code)
# - You want maximum flexibility

# Use ABC when:
# - You want to share implementation code
# - You want to enforce a class hierarchy
# - You need isinstance() checks without @runtime_checkable
```

---

## 8. Data Structures & Collections

### 8.1 Choosing the Right Data Structure

| Need                           | Use                       | Time Complexity               |
| ------------------------------ | ------------------------- | ----------------------------- |
| Ordered sequence, index access | `list`                    | O(1) index, O(n) search       |
| Membership testing             | `set`                     | O(1) average                  |
| Key-value mapping              | `dict`                    | O(1) average                  |
| Immutable sequence             | `tuple`                   | O(1) index, O(n) search       |
| FIFO queue                     | `collections.deque`       | O(1) append/pop both ends     |
| LIFO stack                     | `list` (use append/pop)   | O(1) append/pop               |
| Priority queue                 | `heapq`                   | O(log n) push/pop             |
| Count occurrences              | `collections.Counter`     | O(n) creation, O(1) lookup    |
| Ordered by insertion           | `dict` (3.7+)             | Maintains order               |
| Default values                 | `collections.defaultdict` | O(1) with auto-initialization |

### 8.2 List Best Practices

```python
# ✅ DO: Use list comprehensions for simple transformations
squares = [x ** 2 for x in range(10)]
even_numbers = [x for x in numbers if x % 2 == 0]

# ✅ DO: Use generator expressions for large data (saves memory)
sum_of_squares = sum(x ** 2 for x in range(1_000_000))

# ❌ DON'T: Use list comprehension when generator works
# Bad: Creates entire list in memory
sum(list(x ** 2 for x in range(1_000_000)))

# ✅ DO: Use enumerate() for index + value
for index, item in enumerate(items):
    print(f"{index}: {item}")

# ✅ DO: Use zip() for parallel iteration
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# ✅ DO: Use zip(strict=True) to catch length mismatches (3.10+)
for name, age in zip(names, ages, strict=True):
    print(f"{name} is {age}")

# ✅ DO: Use unpacking
first, *rest = [1, 2, 3, 4, 5]  # first=1, rest=[2,3,4,5]
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2,3,4], last=5

# ❌ DON'T: Modify list while iterating
for item in items:
    if should_remove(item):
        items.remove(item)  # Bug: Skips elements!

# ✅ DO: Create a new list or iterate over a copy
items = [item for item in items if not should_remove(item)]
# Or
for item in items[:]:  # Iterate over a copy
    if should_remove(item):
        items.remove(item)

# ❌ DON'T: Use list for membership testing
if user_id in user_list:  # O(n) - slow for large lists
    ...

# ✅ DO: Use set for membership testing
user_set = set(user_list)  # O(n) once
if user_id in user_set:  # O(1) - fast
    ...
```

### 8.3 Dict Best Practices

```python
# ✅ DO: Use dict.get() for safe access with default
value = my_dict.get("key", "default")

# ✅ DO: Use dict.setdefault() to initialize and get
# Instead of:
if key not in my_dict:
    my_dict[key] = []
my_dict[key].append(item)

# Do:
my_dict.setdefault(key, []).append(item)

# ✅ DO: Use defaultdict for automatic initialization
from collections import defaultdict

# Group items by category
by_category = defaultdict(list)
for item in items:
    by_category[item.category].append(item)

# Count occurrences
counts = defaultdict(int)
for word in words:
    counts[word] += 1

# ✅ DO: Use Counter for counting
from collections import Counter

word_counts = Counter(words)
most_common = word_counts.most_common(10)

# ✅ DO: Use dict comprehensions
squared = {x: x ** 2 for x in range(10)}
filtered = {k: v for k, v in data.items() if v > threshold}

# ✅ DO: Use | for merging dicts (3.9+)
merged = dict1 | dict2  # dict2 values take precedence
dict1 |= dict2  # In-place merge

# ✅ DO: Use .items(), .keys(), .values() for iteration
for key, value in my_dict.items():
    print(f"{key}: {value}")

# ✅ DO: Use dict as switch/case alternative
def get_handler(action: str) -> Callable:
    handlers = {
        "create": handle_create,
        "update": handle_update,
        "delete": handle_delete,
    }
    handler = handlers.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    return handler
```

### 8.4 Set Best Practices

```python
# ✅ DO: Use sets for membership testing
valid_statuses = {"pending", "approved", "rejected"}
if status in valid_statuses:
    ...

# ✅ DO: Use set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2  # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2  # {3, 4}
difference = set1 - set2  # {1, 2}
symmetric_diff = set1 ^ set2  # {1, 2, 5, 6}

# ✅ DO: Use sets to remove duplicates (order not preserved)
unique_items = list(set(items_with_duplicates))

# ✅ DO: Use dict.fromkeys() to remove duplicates (order preserved)
unique_ordered = list(dict.fromkeys(items_with_duplicates))

# ✅ DO: Use frozenset for hashable sets (can be dict keys)
cache_key = frozenset(selected_options)
```

### 8.5 Dataclasses vs NamedTuple vs Pydantic

| Feature            | `dataclass`                       | `NamedTuple`              | `Pydantic`             |
| ------------------ | --------------------------------- | ------------------------- | ---------------------- |
| Mutable            | Default yes, can be `frozen=True` | Always immutable          | Default yes            |
| Validation         | No built-in                       | No built-in               | Yes, extensive         |
| Performance        | Fast                              | Fastest                   | Slower (but improving) |
| JSON serialization | Manual                            | Manual                    | Built-in               |
| Best for           | Internal data, mutable state      | Simple records, dict keys | API boundaries, config |

```python
from dataclasses import dataclass, field
from typing import NamedTuple
from pydantic import BaseModel, EmailStr, Field, field_validator

# Dataclass: General-purpose structured data
@dataclass
class UserProfile:
    id: int
    name: str
    email: str
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validation/transformation after initialization
        self.name = self.name.strip()

# Frozen dataclass: Immutable, can be hashed
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# NamedTuple: Lightweight, immutable, memory-efficient
class Coordinate(NamedTuple):
    latitude: float
    longitude: float
    altitude: float = 0.0

coord = Coordinate(40.7128, -74.0060)
lat, lon, alt = coord  # Unpacking works

# Pydantic: Validation at boundaries (API, config, external data)
class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr  # Validates email format
    age: int = Field(ge=0, le=150)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().title()

# Pydantic automatically validates and converts
user = CreateUserRequest(name="  john doe  ", email="john@example.com", age="25")
print(user.name)  # "John Doe"
print(user.age)   # 25 (converted from string to int)

# Pydantic serialization
json_str = user.model_dump_json()
user_dict = user.model_dump()
```

### 8.6 Collections Module

```python
from collections import deque, defaultdict, Counter, OrderedDict, ChainMap

# deque: Efficient double-ended queue
queue = deque(maxlen=100)  # Auto-removes oldest when full
queue.append("new")
queue.appendleft("first")
item = queue.pop()
item = queue.popleft()

# Use deque for sliding window
def moving_average(values: list[float], window: int) -> list[float]:
    window_deque = deque(maxlen=window)
    result = []
    for value in values:
        window_deque.append(value)
        if len(window_deque) == window:
            result.append(sum(window_deque) / window)
    return result

# defaultdict: Dict with default value factory
graph = defaultdict(list)  # Default is empty list
graph["a"].append("b")
graph["a"].append("c")

# Counter: Count hashable objects
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(counter["apple"])  # 3
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# Counter arithmetic
counter1 = Counter(a=3, b=1)
counter2 = Counter(a=1, b=2)
print(counter1 + counter2)  # Counter({'a': 4, 'b': 3})
print(counter1 - counter2)  # Counter({'a': 2}) - drops zero/negative

# ChainMap: Search multiple dicts as one
defaults = {"color": "blue", "size": "medium"}
user_prefs = {"color": "red"}
settings = ChainMap(user_prefs, defaults)
print(settings["color"])  # "red" (from user_prefs)
print(settings["size"])   # "medium" (from defaults)
```

---

## 9. Module Organization & Avoiding Cyclic Dependencies

### 9.1 Import Best Practices

#### Import Order (PEP 8)

```python
# 1. Standard library imports
import os
import sys
from collections import defaultdict
from pathlib import Path

# 2. Third-party imports
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# 3. Local application imports
from myapp.models import User
from myapp.services import UserService
from . import utils  # Relative import within package
from .config import settings
```

#### Import Styles

```python
# ✅ DO: Import specific names when using few items
from datetime import datetime, timedelta
from pathlib import Path

# ✅ DO: Import module when using many items or avoiding conflicts
import datetime
import json

timestamp = datetime.datetime.now()
data = json.loads(text)

# ❌ DON'T: Use wildcard imports (pollutes namespace, unclear dependencies)
from module import *

# ❌ DON'T: Import inside functions unless necessary
def process():
    import heavy_module  # Delays loading, but hides dependency

# ✅ DO: Import at top unless there's a circular dependency or optional dependency
# Exception: Expensive imports that are rarely used
def rare_operation():
    import pandas as pd  # Only import when needed
    return pd.DataFrame(data)
```

### 9.2 Preventing Cyclic Dependencies

Cyclic dependencies occur when module A imports module B, and module B imports module A (directly or through a chain).

#### Strategy 1: Move Common Code to a Third Module

```python
# ❌ Cyclic dependency
# models.py
from services import UserService  # Imports services
class User: ...

# services.py
from models import User  # Imports models -> CYCLE!
class UserService: ...

# ✅ Fix: Extract shared code to a base module
# base.py (no imports from models or services)
class UserBase:
    id: int
    email: str

# models.py
from base import UserBase
class User(UserBase):
    # Additional fields
    ...

# services.py
from base import UserBase  # No cycle!
class UserService:
    def get_user(self, user_id: int) -> UserBase: ...
```

#### Strategy 2: Use TYPE_CHECKING for Type Hints

```python
from typing import TYPE_CHECKING

# This block only runs during type checking, not at runtime
if TYPE_CHECKING:
    from services import UserService

class User:
    def __init__(self, service: "UserService"):  # String annotation
        self.service = service
```

#### Strategy 3: Import at Function Level

```python
# models.py
class User:
    def get_service(self):
        # Import inside method to break cycle
        from services import UserService
        return UserService(self)
```

#### Strategy 4: Use Dependency Injection

```python
# Instead of importing UserRepository directly in UserService,
# inject it as a dependency

# interfaces.py
from typing import Protocol

class UserRepositoryProtocol(Protocol):
    def get_by_id(self, user_id: int) -> "User": ...

# services.py
from interfaces import UserRepositoryProtocol

class UserService:
    def __init__(self, repository: UserRepositoryProtocol):
        self._repository = repository  # No direct import of repository module!
```

### 9.3 Package Structure

```text
my_application/
├── pyproject.toml
├── src/
│   └── my_app/
│       ├── __init__.py           # Package initialization
│       ├── main.py               # Entry point
│       ├── config.py             # Settings/configuration
│       │
│       ├── domain/               # Business logic (pure Python, no frameworks)
│       │   ├── __init__.py
│       │   ├── models.py         # Domain entities
│       │   ├── services.py       # Business logic
│       │   └── interfaces.py     # Protocols/ABCs
│       │
│       ├── infrastructure/       # External integrations
│       │   ├── __init__.py
│       │   ├── database.py       # Database connections
│       │   ├── cache.py          # Caching
│       │   └── external_api.py   # Third-party API clients
│       │
│       ├── api/                  # Web API layer (FastAPI, Flask, etc.)
│       │   ├── __init__.py
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── users.py
│       │   │   └── orders.py
│       │   ├── schemas.py        # Pydantic models for API
│       │   └── dependencies.py   # FastAPI dependencies
│       │
│       └── utils/                # Shared utilities
│           ├── __init__.py
│           └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── unit/
│   │   └── test_services.py
│   └── integration/
│       └── test_api.py
│
└── scripts/                      # CLI tools, migrations, etc.
    └── seed_database.py
```

#### The `__init__.py` File

```python
# my_app/__init__.py

# Control what's exported with __all__
__all__ = ["UserService", "User", "settings"]

# Convenient re-exports
from .domain.models import User
from .domain.services import UserService
from .config import settings

# Version
__version__ = "1.0.0"
```

---

## 10. Error Handling & Exceptions

### 10.1 Exception Best Practices

#### Catch Specific Exceptions

```python
# ❌ DON'T: Catch all exceptions (hides bugs)
try:
    result = do_something()
except Exception:
    pass  # Silently ignores ALL errors, including bugs!

# ❌ DON'T: Bare except (even worse, catches KeyboardInterrupt, SystemExit)
try:
    result = do_something()
except:
    pass

# ✅ DO: Catch specific exceptions
try:
    result = do_something()
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
    result = default_value
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    raise ServiceUnavailableError("Cannot connect to service") from e
```

#### Create Custom Exceptions

```python
# Base exception for your application
class AppError(Exception):
    """Base exception for the application."""
    pass

# Specific exceptions inherit from base
class ValidationError(AppError):
    """Raised when input validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class NotFoundError(AppError):
    """Raised when a resource is not found."""
    def __init__(self, resource: str, identifier: str | int):
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} not found: {identifier}")

class AuthenticationError(AppError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(AppError):
    """Raised when user lacks permission."""
    pass

# Usage
def get_user(user_id: int) -> User:
    user = repository.get_by_id(user_id)
    if user is None:
        raise NotFoundError("User", user_id)
    return user

# Catching
try:
    user = get_user(123)
except NotFoundError as e:
    print(f"Could not find {e.resource}: {e.identifier}")
```

#### Exception Chaining

```python
# ✅ DO: Use 'from' to chain exceptions (preserves traceback)
try:
    data = json.loads(raw_data)
except json.JSONDecodeError as e:
    raise ValidationError("body", "Invalid JSON") from e

# ✅ DO: Use 'from None' to suppress original exception (when irrelevant)
try:
    value = int(user_input)
except ValueError:
    raise ValidationError("age", "Must be a number") from None
```

### 10.2 Context Managers

**When to use:** Resource management (files, connections, locks), setup/teardown.

```python
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

# ✅ DO: Use context managers for resource management
with open("file.txt") as f:
    content = f.read()
# File automatically closed

# Creating context managers - function style
@contextmanager
def database_transaction(db: Database) -> Generator[Transaction, None, None]:
    """Context manager for database transactions."""
    transaction = db.begin_transaction()
    try:
        yield transaction
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise

# Usage
with database_transaction(db) as txn:
    txn.execute("INSERT INTO users ...")
    txn.execute("UPDATE accounts ...")
# Auto-commits on success, rollbacks on exception

# Creating context managers - class style
class Timer:
    """Context manager to measure execution time."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0
    
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

# Usage
with Timer("database query"):
    results = db.execute(query)

# Async context manager
@asynccontextmanager
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async context manager for HTTP client."""
    client = httpx.AsyncClient()
    try:
        yield client
    finally:
        await client.aclose()

# Usage
async with http_client() as client:
    response = await client.get("https://api.example.com")
```

---

## 11. Testing Best Practices

### 11.1 Test Organization

```text
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Unit tests (no external dependencies)
│   ├── __init__.py
│   ├── test_models.py
│   └── test_services.py
├── integration/          # Integration tests (with real dependencies)
│   ├── __init__.py
│   └── test_api.py
└── e2e/                  # End-to-end tests
    └── test_workflows.py
```

### 11.2 Writing Tests with pytest

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from my_app.services import UserService
from my_app.models import User

# Basic test
def test_user_creation():
    """Test that users are created with correct attributes."""
    user = User(id=1, email="test@example.com", name="Test")
    
    assert user.id == 1
    assert user.email == "test@example.com"
    assert user.name == "Test"

# Test with fixtures
@pytest.fixture
def user_service(mock_repository: Mock) -> UserService:
    """Fixture providing a UserService with mocked repository."""
    return UserService(repository=mock_repository)

@pytest.fixture
def mock_repository() -> Mock:
    """Fixture providing a mock repository."""
    return Mock()

def test_get_user_returns_user(user_service: UserService, mock_repository: Mock):
    """Test that get_user returns the user from repository."""
    expected_user = User(id=1, email="test@example.com", name="Test")
    mock_repository.get_by_id.return_value = expected_user
    
    result = user_service.get_user(1)
    
    assert result == expected_user
    mock_repository.get_by_id.assert_called_once_with(1)

def test_get_user_raises_not_found(user_service: UserService, mock_repository: Mock):
    """Test that get_user raises NotFoundError when user doesn't exist."""
    mock_repository.get_by_id.return_value = None
    
    with pytest.raises(NotFoundError) as exc_info:
        user_service.get_user(999)
    
    assert exc_info.value.resource == "User"
    assert exc_info.value.identifier == 999

# Parametrized tests
@pytest.mark.parametrize("email,is_valid", [
    ("user@example.com", True),
    ("user@sub.example.com", True),
    ("invalid", False),
    ("@example.com", False),
    ("user@", False),
    ("", False),
])
def test_email_validation(email: str, is_valid: bool):
    """Test email validation with various inputs."""
    assert validate_email(email) == is_valid

# Async tests
@pytest.mark.asyncio
async def test_async_fetch_user(user_service: UserService):
    """Test async user fetching."""
    user = await user_service.fetch_user_async(1)
    assert user.id == 1

# Testing exceptions with context
def test_invalid_age_raises_validation_error():
    """Test that negative age raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        create_user(name="Test", age=-1)
    
    assert "age" in str(exc_info.value)

# Mocking external dependencies
@patch("my_app.services.requests.get")
def test_external_api_call(mock_get: Mock):
    """Test external API integration."""
    mock_get.return_value.json.return_value = {"data": "test"}
    mock_get.return_value.status_code = 200
    
    result = fetch_external_data()
    
    assert result == {"data": "test"}
    mock_get.assert_called_once()
```

### 11.3 Testing Guidelines

| Do                                | Don't                         |
| --------------------------------- | ----------------------------- |
| Test behavior, not implementation | Test private methods directly |
| Use descriptive test names        | Use `test_1`, `test_2`        |
| One assertion concept per test    | Multiple unrelated assertions |
| Mock external dependencies        | Mock everything               |
| Test edge cases                   | Only test happy path          |
| Keep tests fast                   | Have slow unit tests          |
| Use fixtures for setup            | Repeat setup in each test     |

---

## 12. Project Structure & Tooling

### 12.1 pyproject.toml (Central Configuration)

```toml
[project]
name = "my-application"
version = "1.0.0"
description = "My Python application"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"},
]
dependencies = [
    "fastapi>=0.100.0",
    "pydantic>=2.0",
    "sqlalchemy>=2.0",
    "httpx>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1.0",
]

[project.scripts]
my-app = "my_app.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Ruff configuration
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # Pyflakes
    "I",     # isort
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "UP",    # pyupgrade
    "ARG",   # flake8-unused-arguments
    "SIM",   # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["my_app"]

# MyPy configuration
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# Pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"

# Coverage configuration
[tool.coverage.run]
source = ["src/my_app"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### 12.2 Modern Toolchain

| Tool         | Purpose                 | Command                         |
| ------------ | ----------------------- | ------------------------------- |
| `uv`         | Fast package management | `uv pip install`, `uv venv`     |
| `ruff`       | Linting + formatting    | `ruff check .`, `ruff format .` |
| `mypy`       | Type checking           | `mypy src/`                     |
| `pytest`     | Testing                 | `pytest tests/`                 |
| `pre-commit` | Git hooks               | `pre-commit run --all-files`    |

#### Pre-commit Configuration (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.0
          - types-requests

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## 13. Performance Considerations

### 13.1 Common Optimizations

```python
# ✅ Use generators for large data
def process_large_file(path: str):
    with open(path) as f:
        for line in f:  # Reads one line at a time
            yield process_line(line)

# ❌ Don't load everything into memory
def process_large_file_bad(path: str):
    with open(path) as f:
        lines = f.readlines()  # Loads entire file into memory!
        return [process_line(line) for line in lines]

# ✅ Use set for membership testing
valid_ids = set(fetch_valid_ids())  # O(n) once
for item in items:
    if item.id in valid_ids:  # O(1) per check
        process(item)

# ❌ Don't use list for membership testing
valid_ids = fetch_valid_ids()  # Returns list
for item in items:
    if item.id in valid_ids:  # O(n) per check!
        process(item)

# ✅ Use str.join() for string concatenation
result = "".join(strings)  # O(n)

# ❌ Don't concatenate in a loop
result = ""
for s in strings:
    result += s  # O(n²) due to string immutability

# ✅ Use list.append() + join vs string concatenation
parts = []
for item in items:
    parts.append(str(item))
result = ", ".join(parts)

# ✅ Use __slots__ for memory-efficient classes (many instances)
class Point:
    __slots__ = ("x", "y")
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# ✅ Use functools.lru_cache for expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    # Result is cached
    return sum(i ** 2 for i in range(n))

# ✅ Use functools.cache for unlimited cache (3.9+)
from functools import cache

@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### 13.2 Async Best Practices

```python
import asyncio
from collections.abc import Coroutine
from typing import Any

# ✅ DO: Use async for I/O-bound operations
async def fetch_user_data(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()

# ✅ DO: Use TaskGroup for concurrent operations (3.11+)
async def fetch_dashboard(user_id: int) -> Dashboard:
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user(user_id))
        orders_task = tg.create_task(fetch_orders(user_id))
        prefs_task = tg.create_task(fetch_preferences(user_id))
    
    return Dashboard(
        user=user_task.result(),
        orders=orders_task.result(),
        preferences=prefs_task.result(),
    )

# ✅ DO: Use asyncio.timeout() for timeouts (3.11+)
async def fetch_with_timeout(url: str) -> dict:
    async with asyncio.timeout(10.0):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()

# ✅ DO: Use semaphores to limit concurrency
async def fetch_all_users(user_ids: list[int]) -> list[User]:
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
    
    async def fetch_with_limit(user_id: int) -> User:
        async with semaphore:
            return await fetch_user(user_id)
    
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_with_limit(uid)) for uid in user_ids]
    
    return [task.result() for task in tasks]

# ❌ DON'T: Mix sync and async improperly
async def bad_example():
    # This blocks the event loop!
    time.sleep(1)  # Should be: await asyncio.sleep(1)
    
    # This also blocks!
    requests.get(url)  # Should use: httpx.AsyncClient

# ❌ DON'T: Create tasks without awaiting them
async def fire_and_forget_bad():
    asyncio.create_task(background_work())  # Task might be garbage collected!

# ✅ DO: Keep references to background tasks
background_tasks = set()

async def fire_and_forget():
    task = asyncio.create_task(background_work())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

---

## 14. Quick Reference Checklists

### Code Review Checklist

- [ ] **Types:** All function signatures have type hints
- [ ] **Names:** Variables, functions, classes have clear, descriptive names
- [ ] **Size:** Functions are focused and under 50 lines
- [ ] **SRP:** Each class/function has a single responsibility
- [ ] **DRY:** No copy-pasted code blocks
- [ ] **Errors:** Specific exceptions caught and handled appropriately
- [ ] **Resources:** Files/connections use context managers
- [ ] **Tests:** New code has corresponding tests
- [ ] **Docs:** Public APIs have docstrings

### New Project Checklist

- [ ] `pyproject.toml` configured with dependencies and tool settings
- [ ] `src/` layout implemented
- [ ] `ruff` configured for linting and formatting
- [ ] `mypy` configured in strict mode
- [ ] `pytest` set up with fixtures in `conftest.py`
- [ ] `pre-commit` hooks installed
- [ ] `.gitignore` includes `__pycache__`, `.mypy_cache`, `.pytest_cache`, etc.
- [ ] README with setup instructions

### Modern Python Feature Adoption

| Feature                    | Min Version | Replaces                     |
| -------------------------- | ----------- | ---------------------------- |
| `match`/`case`             | 3.10        | Complex `if`/`elif` chains   |
| `int \| str` union syntax  | 3.10        | `Union[int, str]`            |
| `list[int]` generic syntax | 3.9         | `List[int]` from typing      |
| `TaskGroup`                | 3.11        | `asyncio.gather()`           |
| `asyncio.timeout()`        | 3.11        | `asyncio.wait_for()`         |
| `Self` type                | 3.11        | `TypeVar` bound to class     |
| `ExceptionGroup`           | 3.11        | Manual exception aggregation |
| `type` alias syntax        | 3.12        | `TypeAlias` annotation       |
| `[T]` generic syntax       | 3.12        | `TypeVar` + `Generic`        |
| `@override`                | 3.12        | Manual verification          |

---

## Summary

Writing high-quality Python code requires:

1. **Applying Design Principles:** SOLID, DRY, KISS, YAGNI keep code maintainable
2. **Using Clean Code Practices:** Good names, small functions, clear intent
3. **Leveraging Modern Python:** Type hints, pattern matching, async best practices
4. **Choosing Right Data Structures:** Understand time complexity, use appropriate types
5. **Organizing Code Properly:** Avoid cycles, use proper module structure
6. **Testing Thoroughly:** Use pytest, fixtures, parametrization
7. **Using Modern Tools:** ruff, mypy, uv, pre-commit

The goal is always code that is:
- **Readable:** Anyone can understand it
- **Maintainable:** Easy to modify and extend
- **Correct:** Does what it's supposed to do
- **Efficient:** Doesn't waste resources unnecessarily