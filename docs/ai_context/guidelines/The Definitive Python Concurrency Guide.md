# The Definitive Python Concurrency Guide
## Threading, Asyncio, and Everything In Between

---

# Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [Python Threading In-Depth](#2-python-threading-in-depth)
3. [Python Asyncio In-Depth](#3-python-asyncio-in-depth)
4. [I/O Blocking: The Complete Reference](#4-io-blocking-the-complete-reference)
5. [Comprehensive Comparison](#5-comprehensive-comparison)
6. [Combining Threading and Asyncio](#6-combining-threading-and-asyncio)
7. [Real-World Patterns and Recipes](#7-real-world-patterns-and-recipes)
8. [Quick Reference Cheatsheets](#8-quick-reference-cheatsheets)

---

# 1. Foundational Concepts

## 1.1 Concurrency vs Parallelism

```
CONCURRENCY (Dealing with multiple things at once)
┌─────────────────────────────────────────────────────────┐
│  Task A: ████░░░░████░░░░████                           │
│  Task B: ░░░░████░░░░████░░░░████                       │
│                                                         │
│  Single core switching between tasks                    │
└─────────────────────────────────────────────────────────┘

PARALLELISM (Doing multiple things at once)
┌─────────────────────────────────────────────────────────┐
│  Core 1 - Task A: ████████████████                      │
│  Core 2 - Task B: ████████████████                      │
│                                                         │
│  Multiple cores executing simultaneously                │
└─────────────────────────────────────────────────────────┘
```

## 1.2 The Global Interpreter Lock (GIL)

The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode at once.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE GIL VISUALIZED                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Thread 1: ████████░░░░░░░░████████░░░░░░░░████████           │
│   Thread 2: ░░░░░░░░████████░░░░░░░░████████░░░░░░░░           │
│                                                                 │
│   ████ = Holding GIL (executing Python code)                   │
│   ░░░░ = Waiting for GIL                                       │
│                                                                 │
│   Key: Only ONE thread holds the GIL at any moment             │
└─────────────────────────────────────────────────────────────────┘
```

**When is the GIL Released?**

```python
# GIL is RELEASED during:
# 1. I/O Operations
data = file.read()        # GIL released while waiting for disk
response = socket.recv()  # GIL released while waiting for network

# 2. Certain C Extensions
import numpy as np
result = np.dot(a, b)     # GIL released during computation

# 3. time.sleep()
time.sleep(1)             # GIL released during sleep

# GIL is HELD during:
# Pure Python computation
total = sum(range(1000000))  # GIL held throughout
```

## 1.3 Types of Workloads

```
┌────────────────────────────────────────────────────────────────┐
│                    WORKLOAD CLASSIFICATION                     │
├──────────────────────┬─────────────────────────────────────────┤
│                      │                                         │
│    CPU-BOUND         │    I/O-BOUND                            │
│    ──────────        │    ────────                             │
│                      │                                         │
│  • Math calculations │  • Network requests                     │
│  • Data processing   │  • File read/write                      │
│  • Image processing  │  • Database queries                     │
│  • Encryption        │  • User input                           │
│  • Compression       │  • API calls                            │
│  • ML training       │  • Web scraping                         │
│                      │                                         │
│  Bottleneck: CPU     │  Bottleneck: Waiting                    │
│                      │                                         │
│  Solution:           │  Solution:                              │
│  multiprocessing     │  threading OR asyncio                   │
│                      │                                         │
└──────────────────────┴─────────────────────────────────────────┘
```

---

# 2. Python Threading In-Depth

## 2.1 What is Threading?

Threading provides **preemptive multitasking** where the operating system scheduler decides when to switch between threads. Each thread shares the same memory space within a process.

```
┌─────────────────────────────────────────────────────────────┐
│                      PROCESS                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SHARED MEMORY SPACE                    │    │
│  │   • Global variables                                │    │
│  │   • Heap (objects, data structures)                 │    │
│  │   • Code segment                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Thread 1 │  │ Thread 2 │  │ Thread 3 │                   │
│  │──────────│  │──────────│  │──────────│                   │
│  │ Stack    │  │ Stack    │  │ Stack    │                   │
│  │ Registers│  │ Registers│  │ Registers│                   │
│  │ PC       │  │ PC       │  │ PC       │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 Thread Lifecycle

```
                    ┌─────────┐
                    │   NEW   │
                    └────┬────┘
                         │ start()
                         ▼
┌──────────┐       ┌─────────┐       ┌────────────┐
│ BLOCKED  │◄─────►│ RUNNABLE│◄─────►│  RUNNING   │
│(waiting) │       │ (ready) │       │            │
└──────────┘       └─────────┘       └─────┬──────┘
     ▲                                      │
     │              ┌─────────┐             │
     └──────────────│ TIMED   │◄────────────┘
                    │ WAITING │   sleep()/wait()
                    └────┬────┘
                         │
                         ▼
                   ┌──────────┐
                   │TERMINATED│
                   └──────────┘
```

## 2.3 Basic Threading Operations

```python
import threading
import time
from typing import Any

# ═══════════════════════════════════════════════════════════════
# METHOD 1: Using Thread with target function
# ═══════════════════════════════════════════════════════════════

def worker(name: str, duration: float) -> None:
    """Simple worker function."""
    thread_id = threading.current_thread().name
    print(f"[{thread_id}] {name} starting, will work for {duration}s")
    time.sleep(duration)  # Simulate work
    print(f"[{thread_id}] {name} finished")

# Create threads
t1 = threading.Thread(target=worker, args=("Task-A", 2))
t2 = threading.Thread(target=worker, args=("Task-B", 1))

# Start threads
t1.start()
t2.start()

# Wait for completion
t1.join()
t2.join()

print("All threads completed")


# ═══════════════════════════════════════════════════════════════
# METHOD 2: Subclassing Thread
# ═══════════════════════════════════════════════════════════════

class DataProcessor(threading.Thread):
    """Custom thread class for data processing."""
    
    def __init__(self, data: list, name: str = None):
        super().__init__(name=name)
        self.data = data
        self.result = None
        self._stop_event = threading.Event()
    
    def run(self) -> None:
        """Main thread execution method."""
        print(f"[{self.name}] Processing {len(self.data)} items")
        
        processed = []
        for i, item in enumerate(self.data):
            if self._stop_event.is_set():
                print(f"[{self.name}] Stopped early at item {i}")
                break
            
            # Simulate processing
            processed.append(item * 2)
            time.sleep(0.1)
        
        self.result = processed
        print(f"[{self.name}] Completed processing")
    
    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()


# Usage
processor = DataProcessor([1, 2, 3, 4, 5], name="Processor-1")
processor.start()
processor.join()
print(f"Result: {processor.result}")


# ═══════════════════════════════════════════════════════════════
# METHOD 3: Daemon Threads
# ═══════════════════════════════════════════════════════════════

def background_monitor():
    """Background task that runs until main program exits."""
    while True:
        print("[Monitor] System check...")
        time.sleep(1)

# Daemon thread automatically stops when main program exits
monitor = threading.Thread(target=background_monitor, daemon=True)
monitor.start()

# Main program continues...
time.sleep(3)
print("Main program ending - daemon will stop automatically")
```

## 2.4 Thread Synchronization Primitives

### 2.4.1 Lock (Mutex)

```python
import threading
import time

# ═══════════════════════════════════════════════════════════════
# LOCK - Basic mutual exclusion
# ═══════════════════════════════════════════════════════════════

class BankAccount:
    """Thread-safe bank account using Lock."""
    
    def __init__(self, balance: float = 0):
        self._balance = balance
        self._lock = threading.Lock()
    
    @property
    def balance(self) -> float:
        return self._balance
    
    def deposit(self, amount: float) -> None:
        with self._lock:  # Acquire lock automatically
            current = self._balance
            time.sleep(0.001)  # Simulate processing delay
            self._balance = current + amount
    
    def withdraw(self, amount: float) -> bool:
        with self._lock:
            if self._balance >= amount:
                current = self._balance
                time.sleep(0.001)
                self._balance = current - amount
                return True
            return False
    
    def transfer(self, other: 'BankAccount', amount: float) -> bool:
        """Transfer money to another account (demonstrates lock ordering)."""
        # Always acquire locks in consistent order to prevent deadlock
        first, second = sorted([self, other], key=id)
        
        with first._lock:
            with second._lock:
                if self._balance >= amount:
                    self._balance -= amount
                    other._balance += amount
                    return True
                return False


# ═══════════════════════════════════════════════════════════════
# DEMONSTRATION: Why locks are necessary
# ═══════════════════════════════════════════════════════════════

def demonstrate_race_condition():
    """Show what happens without proper synchronization."""
    
    class UnsafeCounter:
        def __init__(self):
            self.count = 0
        
        def increment(self):
            current = self.count
            time.sleep(0.0001)  # This tiny delay exposes the race condition
            self.count = current + 1
    
    class SafeCounter:
        def __init__(self):
            self.count = 0
            self._lock = threading.Lock()
        
        def increment(self):
            with self._lock:
                current = self.count
                time.sleep(0.0001)
                self.count = current + 1
    
    def run_increments(counter, n):
        for _ in range(n):
            counter.increment()
    
    # Test unsafe counter
    unsafe = UnsafeCounter()
    threads = [threading.Thread(target=run_increments, args=(unsafe, 100)) 
               for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"Unsafe counter (expected 1000): {unsafe.count}")
    
    # Test safe counter
    safe = SafeCounter()
    threads = [threading.Thread(target=run_increments, args=(safe, 100)) 
               for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"Safe counter (expected 1000): {safe.count}")


demonstrate_race_condition()
```

### 2.4.2 RLock (Reentrant Lock)

```python
import threading

# ═══════════════════════════════════════════════════════════════
# RLOCK - Can be acquired multiple times by the SAME thread
# ═══════════════════════════════════════════════════════════════

class RecursiveDataStructure:
    """Demonstrates when RLock is necessary."""
    
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()  # RLock allows re-entry
    
    def add_item(self, key: str, value: any) -> None:
        with self._lock:
            self._data[key] = value
    
    def add_items(self, items: dict) -> None:
        with self._lock:
            for key, value in items.items():
                self.add_item(key, value)  # Calls method that also acquires lock
                # With regular Lock, this would DEADLOCK!
    
    def process_recursive(self, depth: int = 0) -> int:
        """Recursive method that needs lock at each level."""
        with self._lock:
            if depth >= 5:
                return depth
            return self.process_recursive(depth + 1)


# ═══════════════════════════════════════════════════════════════
# LOCK vs RLOCK Comparison
# ═══════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────────────────────────────────┐
│                    LOCK vs RLOCK                                │
├────────────────────────┬────────────────────────────────────────┤
│        Lock            │           RLock                        │
├────────────────────────┼────────────────────────────────────────┤
│ Can be acquired once   │ Can be acquired multiple times         │
│ per thread             │ by the SAME thread                     │
├────────────────────────┼────────────────────────────────────────┤
│ Second acquire() by    │ Each acquire() must have matching      │
│ same thread = DEADLOCK │ release()                              │
├────────────────────────┼────────────────────────────────────────┤
│ Slightly faster        │ Slightly slower (tracks owner)         │
├────────────────────────┼────────────────────────────────────────┤
│ Use for: simple        │ Use for: recursive functions,          │
│ critical sections      │ nested method calls                    │
└────────────────────────┴────────────────────────────────────────┘
"""
```

### 2.4.3 Semaphore

```python
import threading
import time
import random

# ═══════════════════════════════════════════════════════════════
# SEMAPHORE - Limit concurrent access to a resource
# ═══════════════════════════════════════════════════════════════

class ConnectionPool:
    """Database connection pool using Semaphore."""
    
    def __init__(self, max_connections: int = 5):
        self._semaphore = threading.Semaphore(max_connections)
        self._connections = []
        self._lock = threading.Lock()
        
        # Pre-create connections
        for i in range(max_connections):
            self._connections.append(f"Connection-{i}")
    
    def get_connection(self) -> str:
        """Acquire a connection from the pool."""
        self._semaphore.acquire()  # Block if no connections available
        
        with self._lock:
            connection = self._connections.pop()
            print(f"[{threading.current_thread().name}] Acquired {connection}")
            return connection
    
    def release_connection(self, connection: str) -> None:
        """Return a connection to the pool."""
        with self._lock:
            self._connections.append(connection)
            print(f"[{threading.current_thread().name}] Released {connection}")
        
        self._semaphore.release()


# ═══════════════════════════════════════════════════════════════
# BOUNDED SEMAPHORE - Prevents releasing more than acquired
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """Rate limiter using BoundedSemaphore."""
    
    def __init__(self, max_requests_per_second: int = 10):
        self._semaphore = threading.BoundedSemaphore(max_requests_per_second)
        self._replenish_thread = threading.Thread(
            target=self._replenish_tokens, 
            daemon=True
        )
        self._replenish_thread.start()
    
    def _replenish_tokens(self):
        """Replenish tokens every second."""
        while True:
            time.sleep(1)
            # Try to release up to max tokens
            for _ in range(10):
                try:
                    self._semaphore.release()
                except ValueError:
                    # BoundedSemaphore prevents over-release
                    break
    
    def acquire(self) -> bool:
        """Try to acquire permission for a request."""
        return self._semaphore.acquire(blocking=False)


# Usage demonstration
def worker(pool: ConnectionPool, worker_id: int):
    """Worker that uses connection pool."""
    conn = pool.get_connection()
    try:
        # Simulate database work
        time.sleep(random.uniform(0.5, 2.0))
    finally:
        pool.release_connection(conn)

# Create pool with 3 connections, but 10 workers
pool = ConnectionPool(max_connections=3)
threads = [
    threading.Thread(target=worker, args=(pool, i), name=f"Worker-{i}")
    for i in range(10)
]

for t in threads: t.start()
for t in threads: t.join()
```

### 2.4.4 Event

```python
import threading
import time

# ═══════════════════════════════════════════════════════════════
# EVENT - Simple thread signaling mechanism
# ═══════════════════════════════════════════════════════════════

class DataPipeline:
    """Pipeline with event-based synchronization."""
    
    def __init__(self):
        self.data = None
        self.data_ready = threading.Event()
        self.processing_done = threading.Event()
        self.shutdown = threading.Event()
    
    def producer(self):
        """Produces data items."""
        for i in range(5):
            if self.shutdown.is_set():
                break
            
            # Produce data
            time.sleep(0.5)
            self.data = f"Data-{i}"
            print(f"[Producer] Created {self.data}")
            
            # Signal data is ready
            self.data_ready.set()
            
            # Wait for processing to complete before producing more
            self.processing_done.wait()
            self.processing_done.clear()
        
        print("[Producer] Finished")
    
    def consumer(self):
        """Consumes data items."""
        while not self.shutdown.is_set():
            # Wait for data with timeout
            if self.data_ready.wait(timeout=1.0):
                print(f"[Consumer] Processing {self.data}")
                time.sleep(0.3)  # Simulate processing
                print(f"[Consumer] Done with {self.data}")
                
                self.data_ready.clear()
                self.processing_done.set()
        
        print("[Consumer] Finished")


# ═══════════════════════════════════════════════════════════════
# EVENT Usage Patterns
# ═══════════════════════════════════════════════════════════════

class WorkerCoordinator:
    """Coordinate multiple workers using events."""
    
    def __init__(self, num_workers: int):
        self.start_event = threading.Event()
        self.workers_ready = [threading.Event() for _ in range(num_workers)]
        self.all_done = threading.Event()
        self.results = [None] * num_workers
    
    def worker(self, worker_id: int):
        """Worker that waits for start signal."""
        print(f"[Worker-{worker_id}] Ready and waiting...")
        self.workers_ready[worker_id].set()
        
        # Wait for start signal
        self.start_event.wait()
        
        print(f"[Worker-{worker_id}] Starting work!")
        time.sleep(0.5 * (worker_id + 1))  # Variable work time
        self.results[worker_id] = f"Result from worker {worker_id}"
        print(f"[Worker-{worker_id}] Done!")
    
    def coordinator(self):
        """Coordinate all workers."""
        # Wait for all workers to be ready
        print("[Coordinator] Waiting for all workers...")
        for event in self.workers_ready:
            event.wait()
        
        print("[Coordinator] All workers ready! Starting in 1 second...")
        time.sleep(1)
        
        # Signal all workers to start simultaneously
        self.start_event.set()
        print("[Coordinator] GO!")


# Usage
coord = WorkerCoordinator(5)
threads = [threading.Thread(target=coord.worker, args=(i,)) for i in range(5)]
threads.append(threading.Thread(target=coord.coordinator))

for t in threads: t.start()
for t in threads: t.join()

print(f"Results: {coord.results}")
```

### 2.4.5 Condition

```python
import threading
import time
import random
from collections import deque

# ═══════════════════════════════════════════════════════════════
# CONDITION - Complex synchronization with wait/notify
# ═══════════════════════════════════════════════════════════════

class BoundedBuffer:
    """Thread-safe bounded buffer using Condition."""
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer = deque()
        self.condition = threading.Condition()
    
    def put(self, item) -> None:
        """Add item to buffer, wait if full."""
        with self.condition:
            # Wait while buffer is full
            while len(self.buffer) >= self.capacity:
                print(f"[Producer] Buffer full, waiting...")
                self.condition.wait()
            
            self.buffer.append(item)
            print(f"[Producer] Added {item}, buffer size: {len(self.buffer)}")
            
            # Notify consumers that data is available
            self.condition.notify()
    
    def get(self):
        """Remove and return item from buffer, wait if empty."""
        with self.condition:
            # Wait while buffer is empty
            while len(self.buffer) == 0:
                print(f"[Consumer] Buffer empty, waiting...")
                self.condition.wait()
            
            item = self.buffer.popleft()
            print(f"[Consumer] Got {item}, buffer size: {len(self.buffer)}")
            
            # Notify producers that space is available
            self.condition.notify()
            return item


# ═══════════════════════════════════════════════════════════════
# Advanced Condition: notify_all for broadcast
# ═══════════════════════════════════════════════════════════════

class Barrier:
    """Custom barrier implementation using Condition."""
    
    def __init__(self, parties: int):
        self.parties = parties
        self.count = 0
        self.generation = 0
        self.condition = threading.Condition()
    
    def wait(self) -> int:
        """Wait for all parties to reach the barrier."""
        with self.condition:
            gen = self.generation
            self.count += 1
            
            if self.count == self.parties:
                # Last one to arrive - release everyone
                self.count = 0
                self.generation += 1
                self.condition.notify_all()  # Wake ALL waiting threads
                return 0  # Return 0 for the "winner"
            else:
                # Wait for others
                while gen == self.generation:
                    self.condition.wait()
                return self.count


# ═══════════════════════════════════════════════════════════════
# CONDITION: wait_for with predicate (Python 3.2+)
# ═══════════════════════════════════════════════════════════════

class StateManager:
    """Manage state transitions with Condition."""
    
    STATES = ['INIT', 'STARTING', 'RUNNING', 'STOPPING', 'STOPPED']
    
    def __init__(self):
        self._state = 'INIT'
        self._condition = threading.Condition()
    
    @property
    def state(self) -> str:
        return self._state
    
    def set_state(self, new_state: str) -> None:
        """Change state and notify waiters."""
        with self._condition:
            if new_state not in self.STATES:
                raise ValueError(f"Invalid state: {new_state}")
            print(f"State transition: {self._state} -> {new_state}")
            self._state = new_state
            self._condition.notify_all()
    
    def wait_for_state(self, target_state: str, timeout: float = None) -> bool:
        """Wait until specific state is reached."""
        with self._condition:
            # wait_for automatically handles spurious wakeups
            return self._condition.wait_for(
                predicate=lambda: self._state == target_state,
                timeout=timeout
            )
    
    def wait_for_states(self, target_states: list, timeout: float = None) -> bool:
        """Wait until any of the target states is reached."""
        with self._condition:
            return self._condition.wait_for(
                predicate=lambda: self._state in target_states,
                timeout=timeout
            )
```

### 2.4.6 Barrier

```python
import threading
import time
import random

# ═══════════════════════════════════════════════════════════════
# BARRIER - Synchronize a fixed number of threads
# ═══════════════════════════════════════════════════════════════

class ParallelComputation:
    """Demonstrates barrier for phased computation."""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        # Barrier that waits for all workers
        self.barrier = threading.Barrier(num_workers, action=self._phase_complete)
        self.phase = 0
        self.data = [0] * num_workers
    
    def _phase_complete(self):
        """Called once when all threads reach the barrier."""
        self.phase += 1
        print(f"\n{'='*50}")
        print(f"Phase {self.phase} complete! Data: {self.data}")
        print(f"{'='*50}\n")
    
    def worker(self, worker_id: int):
        """Worker that processes data in phases."""
        for phase in range(3):
            # Phase 1: Initialize
            self.data[worker_id] = worker_id * 10
            print(f"[Worker-{worker_id}] Phase 1: Initialized data")
            self.barrier.wait()
            
            # Phase 2: Process (read neighbors' data)
            time.sleep(random.uniform(0.1, 0.5))
            left = self.data[(worker_id - 1) % self.num_workers]
            right = self.data[(worker_id + 1) % self.num_workers]
            self.data[worker_id] = (left + right) // 2
            print(f"[Worker-{worker_id}] Phase 2: Processed data")
            self.barrier.wait()
            
            # Phase 3: Finalize
            self.data[worker_id] += 1
            print(f"[Worker-{worker_id}] Phase 3: Finalized data")
            self.barrier.wait()


# ═══════════════════════════════════════════════════════════════
# BARRIER: Handling broken barriers
# ═══════════════════════════════════════════════════════════════

def barrier_with_error_handling():
    """Demonstrate barrier error handling."""
    
    barrier = threading.Barrier(3, timeout=2.0)
    
    def worker(worker_id: int, should_fail: bool = False):
        try:
            print(f"[Worker-{worker_id}] Approaching barrier...")
            
            if should_fail:
                time.sleep(0.5)
                raise Exception("Simulated failure")
            
            barrier.wait()
            print(f"[Worker-{worker_id}] Passed barrier!")
            
        except threading.BrokenBarrierError:
            print(f"[Worker-{worker_id}] Barrier was broken!")
        except Exception as e:
            print(f"[Worker-{worker_id}] Failed: {e}")
            barrier.abort()  # Break the barrier for all threads
    
    threads = [
        threading.Thread(target=worker, args=(0, False)),
        threading.Thread(target=worker, args=(1, False)),
        threading.Thread(target=worker, args=(2, True)),  # This one fails
    ]
    
    for t in threads: t.start()
    for t in threads: t.join()


barrier_with_error_handling()
```

## 2.5 Thread-Safe Data Structures

```python
import threading
import queue
import time
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════
# QUEUE MODULE - Thread-safe queues
# ═══════════════════════════════════════════════════════════════

class QueueTypes:
    """Demonstration of different queue types."""
    
    @staticmethod
    def fifo_queue_demo():
        """First-In-First-Out queue."""
        q = queue.Queue(maxsize=5)  # maxsize=0 means unlimited
        
        def producer():
            for i in range(10):
                item = f"item-{i}"
                q.put(item)  # Blocks if queue is full
                print(f"[Producer] Put {item}")
                time.sleep(0.1)
            q.put(None)  # Sentinel to signal completion
        
        def consumer():
            while True:
                item = q.get()  # Blocks if queue is empty
                if item is None:
                    break
                print(f"[Consumer] Got {item}")
                q.task_done()  # Signal that item processing is complete
        
        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    
    @staticmethod
    def lifo_queue_demo():
        """Last-In-First-Out queue (stack)."""
        q = queue.LifoQueue()
        
        for i in range(5):
            q.put(i)
        
        while not q.empty():
            print(q.get())  # Prints: 4, 3, 2, 1, 0
    
    @staticmethod
    def priority_queue_demo():
        """Priority queue (lowest value = highest priority)."""
        q = queue.PriorityQueue()
        
        # Items are tuples: (priority, data)
        q.put((3, "Low priority"))
        q.put((1, "High priority"))
        q.put((2, "Medium priority"))
        
        while not q.empty():
            priority, item = q.get()
            print(f"Priority {priority}: {item}")
        # Output:
        # Priority 1: High priority
        # Priority 2: Medium priority
        # Priority 3: Low priority


# ═══════════════════════════════════════════════════════════════
# NON-BLOCKING QUEUE OPERATIONS
# ═══════════════════════════════════════════════════════════════

def non_blocking_queue_ops():
    """Demonstrate non-blocking queue operations."""
    
    q = queue.Queue(maxsize=2)
    
    # put_nowait - raises queue.Full if queue is full
    try:
        q.put_nowait("item1")
        q.put_nowait("item2")
        q.put_nowait("item3")  # Raises queue.Full
    except queue.Full:
        print("Queue is full!")
    
    # get_nowait - raises queue.Empty if queue is empty
    try:
        while True:
            item = q.get_nowait()
            print(f"Got: {item}")
    except queue.Empty:
        print("Queue is empty!")
    
    # put with timeout
    try:
        q.put("item", timeout=1.0)  # Wait up to 1 second
    except queue.Full:
        print("Timed out waiting to put")
    
    # get with timeout
    try:
        item = q.get(timeout=1.0)  # Wait up to 1 second
    except queue.Empty:
        print("Timed out waiting to get")


# ═══════════════════════════════════════════════════════════════
# THREAD-SAFE COUNTER AND COLLECTIONS
# ═══════════════════════════════════════════════════════════════

class ThreadSafeCounter:
    """Thread-safe counter implementation."""
    
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class ThreadSafeDict:
    """Thread-safe dictionary wrapper."""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()  # RLock for nested operations
    
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
    
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
    
    def __contains__(self, key):
        with self._lock:
            return key in self._dict
    
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
    
    def setdefault(self, key, default=None):
        with self._lock:
            return self._dict.setdefault(key, default)
    
    def update_if_exists(self, key, func):
        """Atomically update value if key exists."""
        with self._lock:
            if key in self._dict:
                self._dict[key] = func(self._dict[key])
                return True
            return False
    
    def items(self):
        with self._lock:
            return list(self._dict.items())
```

## 2.6 ThreadPoolExecutor

```python
import concurrent.futures
import threading
import time
import random
from typing import List, Any

# ═══════════════════════════════════════════════════════════════
# BASIC THREADPOOLEXECUTOR USAGE
# ═══════════════════════════════════════════════════════════════

def fetch_url(url: str) -> dict:
    """Simulate fetching a URL."""
    print(f"[{threading.current_thread().name}] Fetching {url}")
    time.sleep(random.uniform(0.5, 2.0))  # Simulate network delay
    return {"url": url, "status": 200, "size": random.randint(1000, 10000)}


def basic_executor_demo():
    """Basic ThreadPoolExecutor patterns."""
    
    urls = [f"https://example.com/page/{i}" for i in range(10)]
    
    # Pattern 1: submit() - returns Future objects
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_url, url): url for url in urls}
        
        # Process as completed
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                print(f"Completed: {url} -> {result['size']} bytes")
            except Exception as e:
                print(f"Failed: {url} -> {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Pattern 2: map() - simpler, maintains order
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_url, urls)
        
        for url, result in zip(urls, results):
            print(f"{url} -> {result['size']} bytes")


# ═══════════════════════════════════════════════════════════════
# ADVANCED EXECUTOR PATTERNS
# ═══════════════════════════════════════════════════════════════

class TaskManager:
    """Advanced task management with ThreadPoolExecutor."""
    
    def __init__(self, max_workers: int = 10):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict = {}
        self._lock = threading.Lock()
    
    def submit(self, task_id: str, fn, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task and track it by ID."""
        future = self.executor.submit(fn, *args, **kwargs)
        
        with self._lock:
            self.futures[task_id] = future
        
        # Add callback to clean up when done
        future.add_done_callback(lambda f: self._task_completed(task_id, f))
        
        return future
    
    def _task_completed(self, task_id: str, future: concurrent.futures.Future):
        """Callback when task completes."""
        try:
            result = future.result()
            print(f"[TaskManager] Task {task_id} completed: {result}")
        except Exception as e:
            print(f"[TaskManager] Task {task_id} failed: {e}")
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            future = self.futures.get(task_id)
            if future:
                return future.cancel()
            return False
    
    def get_status(self, task_id: str) -> str:
        """Get task status."""
        with self._lock:
            future = self.futures.get(task_id)
            if not future:
                return "NOT_FOUND"
            if future.cancelled():
                return "CANCELLED"
            if future.running():
                return "RUNNING"
            if future.done():
                return "COMPLETED"
            return "PENDING"
    
    def wait_all(self, timeout: float = None) -> tuple:
        """Wait for all tasks to complete."""
        with self._lock:
            futures_list = list(self.futures.values())
        
        done, not_done = concurrent.futures.wait(
            futures_list,
            timeout=timeout,
            return_when=concurrent.futures.ALL_COMPLETED
        )
        return done, not_done
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)


# ═══════════════════════════════════════════════════════════════
# HANDLING EXCEPTIONS IN EXECUTOR
# ═══════════════════════════════════════════════════════════════

def exception_handling_demo():
    """Demonstrate exception handling with executors."""
    
    def risky_task(n: int) -> int:
        """Task that might fail."""
        if n % 3 == 0:
            raise ValueError(f"Don't like number {n}")
        time.sleep(0.1)
        return n * 2
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(risky_task, i): i for i in range(10)}
        
        results = []
        errors = []
        
        for future in concurrent.futures.as_completed(futures):
            n = futures[future]
            try:
                result = future.result()
                results.append((n, result))
            except Exception as e:
                errors.append((n, str(e)))
        
        print(f"Successful: {sorted(results)}")
        print(f"Failed: {sorted(errors)}")


# ═══════════════════════════════════════════════════════════════
# EXECUTOR WITH TIMEOUT
# ═══════════════════════════════════════════════════════════════

def timeout_demo():
    """Demonstrate timeout handling."""
    
    def slow_task(duration: float) -> str:
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future = executor.submit(slow_task, 5.0)
        
        try:
            # Wait only 2 seconds for result
            result = future.result(timeout=2.0)
            print(result)
        except concurrent.futures.TimeoutError:
            print("Task took too long!")
            # Note: The task continues running in background
            # Cancel won't work if task already started
            cancelled = future.cancel()
            print(f"Cancel attempted: {cancelled}")


# Run demos
if __name__ == "__main__":
    basic_executor_demo()
    exception_handling_demo()
    timeout_demo()
```

## 2.7 Thread Pros and Cons

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           THREADING PROS & CONS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✅ ADVANTAGES                                                               ║
║  ─────────────                                                               ║
║                                                                              ║
║  1. SIMPLICITY                                                               ║
║     • Familiar sequential programming model                                  ║
║     • No need to rewrite existing blocking code                              ║
║     • Easy to understand execution flow                                      ║
║                                                                              ║
║  2. COMPATIBILITY                                                            ║
║     • Works with ANY Python library (blocking or not)                        ║
║     • Integrates with legacy code easily                                     ║
║     • No special "async-compatible" libraries needed                         ║
║                                                                              ║
║  3. TRUE PREEMPTION                                                          ║
║     • OS handles switching automatically                                     ║
║     • Long-running code won't block other threads                            ║
║     • Good for I/O operations that release the GIL                           ║
║                                                                              ║
║  4. SHARED MEMORY                                                            ║
║     • Efficient data sharing between threads                                 ║
║     • No serialization overhead for communication                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ❌ DISADVANTAGES                                                            ║
║  ──────────────                                                              ║
║                                                                              ║
║  1. GIL LIMITATION                                                           ║
║     • Only one thread executes Python bytecode at a time                     ║
║     • No true parallelism for CPU-bound tasks                                ║
║     • Limits performance gains for computation                               ║
║                                                                              ║
║  2. RACE CONDITIONS                                                          ║
║     • Shared mutable state requires careful synchronization                  ║
║     • Bugs can be intermittent and hard to reproduce                         ║
║     • Debugging is challenging (Heisenbugs)                                  ║
║                                                                              ║
║  3. DEADLOCKS                                                                ║
║     • Multiple locks can cause threads to wait forever                       ║
║     • Requires careful lock ordering                                         ║
║     • Can be hard to detect                                                  ║
║                                                                              ║
║  4. RESOURCE OVERHEAD                                                        ║
║     • Each thread has its own stack (typically 1-8 MB)                       ║
║     • Context switching has CPU overhead                                     ║
║     • Limited scalability (hundreds, not thousands)                          ║
║                                                                              ║
║  5. NON-DETERMINISTIC                                                        ║
║     • Execution order is unpredictable                                       ║
║     • Makes testing difficult                                                ║
║     • Race conditions may only appear under load                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## 2.8 Threading Best Practices

```python
# ═══════════════════════════════════════════════════════════════
# BEST PRACTICES FOR PYTHON THREADING
# ═══════════════════════════════════════════════════════════════

"""
BEST PRACTICE 1: Prefer ThreadPoolExecutor over manual thread management
"""
# ❌ Bad
threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()

# ✅ Good
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(worker, range(10))


"""
BEST PRACTICE 2: Always use context managers for locks
"""
# ❌ Bad - might not release lock if exception occurs
lock.acquire()
do_something()
lock.release()

# ✅ Good - always releases lock
with lock:
    do_something()


"""
BEST PRACTICE 3: Avoid shared mutable state when possible
"""
# ❌ Bad - shared mutable state
class Counter:
    def __init__(self):
        self.count = 0  # Shared, mutable
    
    def increment(self):
        self.count += 1  # Race condition!

# ✅ Good - use thread-local storage or message passing
import queue

def worker(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:
            break
        result = process(item)
        output_queue.put(result)


"""
BEST PRACTICE 4: Use threading.local for thread-specific data
"""
# Thread-local storage
local_data = threading.local()

def worker():
    # Each thread has its own 'connection'
    local_data.connection = create_connection()
    try:
        do_work(local_data.connection)
    finally:
        local_data.connection.close()


"""
BEST PRACTICE 5: Implement proper shutdown mechanisms
"""
class GracefulWorker:
    def __init__(self):
        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self):
        self._thread.start()
    
    def stop(self, timeout=5.0):
        self._shutdown_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("Warning: Worker did not stop gracefully")
    
    def _run(self):
        while not self._shutdown_event.is_set():
            # Check for shutdown periodically
            if self._shutdown_event.wait(timeout=0.1):
                break
            self._do_work()


"""
BEST PRACTICE 6: Consistent lock ordering to prevent deadlocks
"""
# ❌ Bad - potential deadlock
def transfer_bad(from_acc, to_acc, amount):
    with from_acc.lock:
        with to_acc.lock:
            # If two threads call transfer(A,B) and transfer(B,A) simultaneously
            # they may deadlock
            pass

# ✅ Good - consistent ordering
def transfer_good(from_acc, to_acc, amount):
    # Always lock in order of account ID
    first, second = sorted([from_acc, to_acc], key=lambda x: x.id)
    with first.lock:
        with second.lock:
            # Safe from deadlock
            pass


"""
BEST PRACTICE 7: Use timeouts to prevent hanging
"""
# ❌ Bad - may hang forever
lock.acquire()
result = future.result()
event.wait()

# ✅ Good - use timeouts
if not lock.acquire(timeout=5.0):
    raise TimeoutError("Could not acquire lock")

try:
    result = future.result(timeout=30.0)
except TimeoutError:
    handle_timeout()

if not event.wait(timeout=10.0):
    raise TimeoutError("Event not set in time")
```

---

# 3. Python Asyncio In-Depth

## 3.1 What is Asyncio?

Asyncio provides **cooperative multitasking** through an event loop. Tasks voluntarily yield control using `await`, allowing other tasks to run.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASYNCIO EVENT LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│                         ┌───▶│ Task Queue  │◀───┐                           │
│                         │    └──────┬──────┘    │                           │
│                         │           │           │                           │
│                         │           ▼           │                           │
│                    ┌────┴────┐  ┌────────┐  ┌───┴───┐                       │
│                    │ Ready   │  │ Event  │  │ I/O   │                       │
│                    │ Tasks   │─▶│ Loop   │◀─│ Events│                       │
│                    └─────────┘  └───┬────┘  └───────┘                       │
│                                     │                                       │
│              ┌──────────────────────┼──────────────────────┐                │
│              │                      │                      │                │
│              ▼                      ▼                      ▼                │
│        ┌──────────┐          ┌──────────┐          ┌──────────┐             │
│        │  Task A  │          │  Task B  │          │  Task C  │             │
│        │   await  │          │   await  │          │   await  │             │
│        │    ↓     │          │    ↓     │          │    ↓     │             │
│        │  yields  │          │  yields  │          │  yields  │             │
│        └──────────┘          └──────────┘          └──────────┘             │
│                                                                             │
│  Single Thread - Tasks cooperate by yielding at await points                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Core Concepts

### 3.2.1 Coroutines, Tasks, and Futures

```python
import asyncio
from typing import Any

# ═══════════════════════════════════════════════════════════════
# COROUTINES - The building blocks
# ═══════════════════════════════════════════════════════════════

async def simple_coroutine() -> str:
    """A coroutine is defined with 'async def'."""
    await asyncio.sleep(1)  # Yield control to event loop
    return "Done!"

# A coroutine is NOT executed when called - it returns a coroutine object
coro = simple_coroutine()
print(type(coro))  # <class 'coroutine'>

# Must be awaited or scheduled to run
result = asyncio.run(simple_coroutine())


# ═══════════════════════════════════════════════════════════════
# TASKS - Scheduled coroutines
# ═══════════════════════════════════════════════════════════════

async def demonstrate_tasks():
    """Tasks wrap coroutines and schedule them for execution."""
    
    async def worker(name: str, duration: float) -> str:
        print(f"[{name}] Starting...")
        await asyncio.sleep(duration)
        print(f"[{name}] Done!")
        return f"Result from {name}"
    
    # Create tasks - they start executing immediately
    task1 = asyncio.create_task(worker("Task-1", 2))
    task2 = asyncio.create_task(worker("Task-2", 1))
    
    # Task properties
    print(f"Task1 name: {task1.get_name()}")
    print(f"Task1 done: {task1.done()}")
    
    # Wait for tasks
    result1 = await task1
    result2 = await task2
    
    return result1, result2


# ═══════════════════════════════════════════════════════════════
# FUTURES - Low-level awaitable objects
# ═══════════════════════════════════════════════════════════════

async def demonstrate_futures():
    """Futures represent eventual results."""
    
    loop = asyncio.get_event_loop()
    
    # Create a Future manually (rarely needed in application code)
    future = loop.create_future()
    
    async def set_future_result():
        await asyncio.sleep(1)
        future.set_result("Future completed!")
    
    # Schedule the setter
    asyncio.create_task(set_future_result())
    
    # Wait for the future
    result = await future
    print(result)


# ═══════════════════════════════════════════════════════════════
# COROUTINE vs TASK vs FUTURE
# ═══════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COROUTINE vs TASK vs FUTURE                              │
├────────────────┬────────────────────┬───────────────────────────────────────┤
│   Coroutine    │       Task         │            Future                     │
├────────────────┼────────────────────┼───────────────────────────────────────┤
│ • async def    │ • Wraps coroutine  │ • Low-level awaitable                 │
│   function     │ • Immediately      │ • Represents eventual                 │
│ • Lazy - must  │   scheduled        │   result                              │
│   be awaited   │ • Can be cancelled │ • Can be set manually                 │
│ • Returns      │ • Tracks state     │ • Used by Tasks internally            │
│   coroutine    │   (pending, done)  │ • Rarely used directly                │
│   object       │                    │                                       │
├────────────────┼────────────────────┼───────────────────────────────────────┤
│ my_coro()      │ asyncio.create_    │ loop.create_future()                  │
│                │ task(my_coro())    │                                       │
└────────────────┴────────────────────┴───────────────────────────────────────┘
"""
```

### 3.2.2 Running Async Code

```python
import asyncio
import sys

# ═══════════════════════════════════════════════════════════════
# WAYS TO RUN ASYNC CODE
# ═══════════════════════════════════════════════════════════════

async def main():
    """Main async entry point."""
    await asyncio.sleep(1)
    return "Done"


# Method 1: asyncio.run() - Recommended for scripts (Python 3.7+)
# Creates event loop, runs coroutine, closes loop
if __name__ == "__main__":
    result = asyncio.run(main())


# Method 2: Get event loop manually (legacy, pre-3.7)
# Useful when you need more control
loop = asyncio.get_event_loop()
try:
    result = loop.run_until_complete(main())
finally:
    loop.close()


# Method 3: Running in existing loop (for frameworks)
async def app_framework():
    # When already in an async context
    result = await main()


# Method 4: Create new loop (for threading scenarios)
def run_in_new_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════
# EVENT LOOP OPERATIONS
# ═══════════════════════════════════════════════════════════════

async def event_loop_operations():
    """Common event loop operations."""
    
    loop = asyncio.get_running_loop()
    
    # Get loop time (monotonic clock)
    current_time = loop.time()
    print(f"Loop time: {current_time}")
    
    # Schedule callback for later
    def callback(message):
        print(f"Callback: {message}")
    
    # Call soon (next iteration)
    loop.call_soon(callback, "called soon")
    
    # Call later (after delay)
    loop.call_later(0.5, callback, "called after 0.5s")
    
    # Call at specific time
    loop.call_at(loop.time() + 1.0, callback, "called at specific time")
    
    await asyncio.sleep(2)  # Wait for callbacks


# ═══════════════════════════════════════════════════════════════
# DEBUG MODE
# ═══════════════════════════════════════════════════════════════

# Enable debug mode for development
asyncio.run(main(), debug=True)

# Or via environment variable
# PYTHONASYNCIODEBUG=1 python script.py

# Debug mode provides:
# - Warnings for coroutines that weren't awaited
# - Longer stack traces
# - Warnings for slow callbacks (>100ms)
```

### 3.2.3 Concurrent Execution Patterns

```python
import asyncio
import time
from typing import List, Any

# ═══════════════════════════════════════════════════════════════
# PATTERN 1: asyncio.gather() - Run multiple coroutines concurrently
# ═══════════════════════════════════════════════════════════════

async def fetch_data(source: str, delay: float) -> dict:
    """Simulate fetching data from a source."""
    print(f"Fetching from {source}...")
    await asyncio.sleep(delay)
    return {"source": source, "data": f"Data from {source}"}


async def gather_example():
    """Run multiple coroutines and collect all results."""
    
    start = time.time()
    
    # All three run concurrently
    results = await asyncio.gather(
        fetch_data("API-1", 2),
        fetch_data("API-2", 1),
        fetch_data("API-3", 3),
    )
    
    print(f"Time: {time.time() - start:.1f}s")  # ~3s, not 6s
    print(f"Results: {results}")
    return results


async def gather_with_exceptions():
    """Handle exceptions in gather."""
    
    async def maybe_fail(n: int):
        await asyncio.sleep(0.1)
        if n == 2:
            raise ValueError(f"Failed at {n}")
        return n * 10
    
    # return_exceptions=True returns exceptions as results instead of raising
    results = await asyncio.gather(
        maybe_fail(1),
        maybe_fail(2),
        maybe_fail(3),
        return_exceptions=True
    )
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")


# ═══════════════════════════════════════════════════════════════
# PATTERN 2: asyncio.wait() - More control over completion
# ═══════════════════════════════════════════════════════════════

async def wait_example():
    """Different wait strategies."""
    
    async def task(n: int, duration: float):
        await asyncio.sleep(duration)
        if n == 2:
            raise Exception(f"Task {n} failed")
        return f"Task {n} done"
    
    tasks = [
        asyncio.create_task(task(1, 1)),
        asyncio.create_task(task(2, 2)),
        asyncio.create_task(task(3, 3)),
    ]
    
    # Wait for FIRST_COMPLETED
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    print(f"First completed: {len(done)}, Pending: {len(pending)}")
    
    # Wait for FIRST_EXCEPTION
    done, pending = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_EXCEPTION
    )
    print(f"After exception: {len(done)}, Pending: {len(pending)}")
    
    # Wait for ALL_COMPLETED
    done, pending = await asyncio.wait(
        pending,
        return_when=asyncio.ALL_COMPLETED
    )
    print(f"All done: {len(done)}, Pending: {len(pending)}")


# ═══════════════════════════════════════════════════════════════
# PATTERN 3: asyncio.as_completed() - Process as they finish
# ═══════════════════════════════════════════════════════════════

async def as_completed_example():
    """Process results as they become available."""
    
    async def fetch(url: str, delay: float) -> dict:
        await asyncio.sleep(delay)
        return {"url": url, "delay": delay}
    
    tasks = [
        fetch("url1", 3),
        fetch("url2", 1),
        fetch("url3", 2),
    ]
    
    # Process in completion order, not submission order
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Completed: {result}")
    
    # Output order: url2 (1s), url3 (2s), url1 (3s)


# ═══════════════════════════════════════════════════════════════
# PATTERN 4: asyncio.wait_for() - Single task with timeout
# ═══════════════════════════════════════════════════════════════

async def timeout_example():
    """Execute with timeout."""
    
    async def slow_operation():
        await asyncio.sleep(10)
        return "Completed"
    
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")


# ═══════════════════════════════════════════════════════════════
# PATTERN 5: TaskGroup (Python 3.11+) - Structured concurrency
# ═══════════════════════════════════════════════════════════════

async def taskgroup_example():
    """Modern structured concurrency with TaskGroup."""
    
    results = []
    
    async def worker(n: int):
        await asyncio.sleep(n * 0.1)
        results.append(n)
        return n * 10
    
    async with asyncio.TaskGroup() as tg:
        # All tasks created in the group
        task1 = tg.create_task(worker(1))
        task2 = tg.create_task(worker(2))
        task3 = tg.create_task(worker(3))
    
    # All tasks guaranteed complete here
    print(f"Results: {task1.result()}, {task2.result()}, {task3.result()}")
    
    # If any task raises an exception, ALL tasks are cancelled


# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

"""
┌──────────────────┬────────────────────────────────────────────────────────┐
│     Method       │                    Use Case                            │
├──────────────────┼────────────────────────────────────────────────────────┤
│ asyncio.gather() │ Run multiple coroutines, need ALL results in order     │
│                  │ Simple error handling with return_exceptions           │
├──────────────────┼────────────────────────────────────────────────────────┤
│ asyncio.wait()   │ Need control over completion strategy                  │
│                  │ (FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED)      │
├──────────────────┼────────────────────────────────────────────────────────┤
│ as_completed()   │ Process results as they arrive                         │
│                  │ Good for progress updates                              │
├──────────────────┼────────────────────────────────────────────────────────┤
│ wait_for()       │ Single coroutine with timeout                          │
├──────────────────┼────────────────────────────────────────────────────────┤
│ TaskGroup        │ Structured concurrency (3.11+)                         │
│                  │ Automatic cleanup on exceptions                        │
└──────────────────┴────────────────────────────────────────────────────────┘
"""
```

## 3.3 Asyncio Synchronization Primitives

```python
import asyncio
from typing import Any

# ═══════════════════════════════════════════════════════════════
# ASYNCIO LOCK - Mutual exclusion for coroutines
# ═══════════════════════════════════════════════════════════════

class AsyncCache:
    """Thread-safe async cache with Lock."""
    
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def get_or_compute(self, key: str, compute_func) -> Any:
        """Get from cache or compute value."""
        # Check without lock first (optimization)
        if key in self._cache:
            return self._cache[key]
        
        async with self._lock:
            # Double-check after acquiring lock
            if key in self._cache:
                return self._cache[key]
            
            # Compute and cache
            value = await compute_func(key)
            self._cache[key] = value
            return value


async def lock_demo():
    """Demonstrate asyncio.Lock."""
    
    lock = asyncio.Lock()
    shared_resource = []
    
    async def worker(name: str):
        print(f"[{name}] Waiting for lock...")
        
        async with lock:
            print(f"[{name}] Acquired lock")
            shared_resource.append(name)
            await asyncio.sleep(1)  # Other tasks can run, but can't get lock
            print(f"[{name}] Releasing lock")
    
    await asyncio.gather(
        worker("Task-1"),
        worker("Task-2"),
        worker("Task-3"),
    )
    
    print(f"Order: {shared_resource}")


# ═══════════════════════════════════════════════════════════════
# ASYNCIO SEMAPHORE - Limit concurrent access
# ═══════════════════════════════════════════════════════════════

class RateLimitedClient:
    """HTTP client with rate limiting."""
    
    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch(self, url: str) -> dict:
        """Fetch URL with rate limiting."""
        async with self._semaphore:
            print(f"Fetching {url}...")
            await asyncio.sleep(1)  # Simulate request
            return {"url": url, "status": 200}
    
    async def fetch_many(self, urls: list) -> list:
        """Fetch multiple URLs with concurrency limit."""
        tasks = [self.fetch(url) for url in urls]
        return await asyncio.gather(*tasks)


async def semaphore_demo():
    """Demonstrate asyncio.Semaphore."""
    
    # Only 2 tasks can be in critical section at a time
    semaphore = asyncio.Semaphore(2)
    
    async def limited_task(n: int):
        print(f"[Task-{n}] Waiting...")
        async with semaphore:
            print(f"[Task-{n}] In critical section")
            await asyncio.sleep(1)
            print(f"[Task-{n}] Done")
    
    start = asyncio.get_event_loop().time()
    await asyncio.gather(*[limited_task(i) for i in range(6)])
    elapsed = asyncio.get_event_loop().time() - start
    
    print(f"Total time: {elapsed:.1f}s")  # ~3s (6 tasks, 2 at a time)


# ═══════════════════════════════════════════════════════════════
# ASYNCIO EVENT - Signal between coroutines
# ═══════════════════════════════════════════════════════════════

class AsyncWorkflow:
    """Workflow with event-based synchronization."""
    
    def __init__(self):
        self.data_ready = asyncio.Event()
        self.processing_done = asyncio.Event()
        self.data = None
    
    async def producer(self):
        """Produces data."""
        for i in range(3):
            await asyncio.sleep(0.5)
            self.data = f"Data-{i}"
            print(f"[Producer] Created {self.data}")
            
            self.data_ready.set()
            await self.processing_done.wait()
            self.processing_done.clear()
        
        self.data = None
        self.data_ready.set()  # Signal end
    
    async def consumer(self):
        """Consumes data."""
        while True:
            await self.data_ready.wait()
            self.data_ready.clear()
            
            if self.data is None:
                print("[Consumer] No more data")
                break
            
            print(f"[Consumer] Processing {self.data}")
            await asyncio.sleep(0.3)
            self.processing_done.set()


async def event_demo():
    """Demonstrate asyncio.Event."""
    
    event = asyncio.Event()
    
    async def waiter(name: str):
        print(f"[{name}] Waiting for event...")
        await event.wait()
        print(f"[{name}] Event received!")
    
    async def setter():
        await asyncio.sleep(1)
        print("[Setter] Setting event")
        event.set()
    
    await asyncio.gather(
        waiter("Waiter-1"),
        waiter("Waiter-2"),
        waiter("Waiter-3"),
        setter(),
    )


# ═══════════════════════════════════════════════════════════════
# ASYNCIO CONDITION - Complex synchronization
# ═══════════════════════════════════════════════════════════════

class AsyncBoundedBuffer:
    """Bounded buffer using Condition."""
    
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.buffer = []
        self.condition = asyncio.Condition()
    
    async def put(self, item: Any) -> None:
        """Add item, wait if full."""
        async with self.condition:
            while len(self.buffer) >= self.capacity:
                await self.condition.wait()
            
            self.buffer.append(item)
            self.condition.notify()  # Wake one consumer
    
    async def get(self) -> Any:
        """Remove item, wait if empty."""
        async with self.condition:
            while len(self.buffer) == 0:
                await self.condition.wait()
            
            item = self.buffer.pop(0)
            self.condition.notify()  # Wake one producer
            return item


async def condition_demo():
    """Demonstrate asyncio.Condition."""
    
    condition = asyncio.Condition()
    ready = False
    
    async def consumer():
        async with condition:
            print("[Consumer] Waiting for condition...")
            await condition.wait_for(lambda: ready)
            print("[Consumer] Condition met!")
    
    async def producer():
        nonlocal ready
        await asyncio.sleep(1)
        async with condition:
            print("[Producer] Setting condition")
            ready = True
            condition.notify_all()
    
    await asyncio.gather(consumer(), producer())


# ═══════════════════════════════════════════════════════════════
# ASYNCIO QUEUES
# ═══════════════════════════════════════════════════════════════

async def queue_demo():
    """Demonstrate asyncio queues."""
    
    # Standard FIFO queue
    queue = asyncio.Queue(maxsize=5)
    
    async def producer():
        for i in range(10):
            await queue.put(f"item-{i}")
            print(f"[Producer] Put item-{i}")
        await queue.put(None)  # Sentinel
    
    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            print(f"[Consumer] Got {item}")
            await asyncio.sleep(0.1)
            queue.task_done()
    
    await asyncio.gather(producer(), consumer())


async def priority_queue_demo():
    """Async priority queue."""
    
    pq = asyncio.PriorityQueue()
    
    await pq.put((2, "medium priority"))
    await pq.put((1, "high priority"))
    await pq.put((3, "low priority"))
    
    while not pq.empty():
        priority, item = await pq.get()
        print(f"Priority {priority}: {item}")


# ═══════════════════════════════════════════════════════════════
# IMPORTANT: Asyncio primitives are NOT thread-safe!
# ═══════════════════════════════════════════════════════════════

"""
⚠️ WARNING: asyncio synchronization primitives are designed for
   coordinating coroutines WITHIN the same event loop.
   
   They are NOT safe to use across threads!

   For thread + asyncio scenarios, use:
   - loop.call_soon_threadsafe()
   - asyncio.run_coroutine_threadsafe()
   - janus (third-party library for dual queues)
"""
```

## 3.4 Async Context Managers and Iterators

```python
import asyncio
from typing import AsyncIterator, Any

# ═══════════════════════════════════════════════════════════════
# ASYNC CONTEXT MANAGERS
# ═══════════════════════════════════════════════════════════════

class AsyncDatabaseConnection:
    """Async context manager for database connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        """Async enter - called with 'async with'."""
        print(f"Connecting to {self.connection_string}...")
        await asyncio.sleep(0.5)  # Simulate connection time
        self.connection = {"connected": True, "url": self.connection_string}
        print("Connected!")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit - cleanup."""
        print("Closing connection...")
        await asyncio.sleep(0.2)  # Simulate cleanup
        self.connection = None
        print("Connection closed!")
        return False  # Don't suppress exceptions
    
    async def query(self, sql: str) -> list:
        """Execute a query."""
        if not self.connection:
            raise RuntimeError("Not connected")
        await asyncio.sleep(0.1)  # Simulate query
        return [{"result": f"Data for: {sql}"}]


# Using contextlib for async context managers
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_resource(name: str):
    """Async context manager using decorator."""
    print(f"Acquiring {name}...")
    await asyncio.sleep(0.1)
    resource = {"name": name, "acquired": True}
    try:
        yield resource
    finally:
        print(f"Releasing {name}...")
        await asyncio.sleep(0.1)


async def context_manager_demo():
    """Demonstrate async context managers."""
    
    # Using class-based
    async with AsyncDatabaseConnection("postgresql://localhost/db") as db:
        results = await db.query("SELECT * FROM users")
        print(results)
    
    # Using decorator-based
    async with async_resource("file_lock") as resource:
        print(f"Using {resource}")


# ═══════════════════════════════════════════════════════════════
# ASYNC ITERATORS
# ═══════════════════════════════════════════════════════════════

class AsyncCounter:
    """Async iterator example."""
    
    def __init__(self, start: int, end: int):
        self.current = start
        self.end = end
    
    def __aiter__(self):
        """Return async iterator."""
        return self
    
    async def __anext__(self):
        """Get next value asynchronously."""
        if self.current >= self.end:
            raise StopAsyncIteration
        
        await asyncio.sleep(0.1)  # Simulate async operation
        value = self.current
        self.current += 1
        return value


class AsyncDatabaseCursor:
    """Async iterator for database results."""
    
    def __init__(self, query: str, batch_size: int = 100):
        self.query = query
        self.batch_size = batch_size
        self.offset = 0
        self.buffer = []
        self.exhausted = False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.buffer and not self.exhausted:
            # Fetch next batch
            self.buffer = await self._fetch_batch()
            if not self.buffer:
                self.exhausted = True
        
        if not self.buffer:
            raise StopAsyncIteration
        
        return self.buffer.pop(0)
    
    async def _fetch_batch(self) -> list:
        """Fetch batch of results."""
        await asyncio.sleep(0.1)  # Simulate database call
        
        # Simulate some data
        if self.offset >= 5:
            return []
        
        batch = [f"Row-{self.offset + i}" for i in range(self.batch_size)]
        self.offset += self.batch_size
        return batch[:3]  # Return only 3 for demo


async def iterator_demo():
    """Demonstrate async iterators."""
    
    # Using async for
    async for value in AsyncCounter(0, 5):
        print(f"Count: {value}")
    
    # Iterate database results
    cursor = AsyncDatabaseCursor("SELECT * FROM table")
    async for row in cursor:
        print(f"Row: {row}")


# ═══════════════════════════════════════════════════════════════
# ASYNC GENERATORS
# ═══════════════════════════════════════════════════════════════

async def async_range(start: int, end: int, delay: float = 0.1):
    """Async generator function."""
    for i in range(start, end):
        await asyncio.sleep(delay)
        yield i


async def fetch_pages(base_url: str, max_pages: int = 10):
    """Async generator for paginated API."""
    page = 1
    while page <= max_pages:
        await asyncio.sleep(0.2)  # Simulate API call
        
        # Simulate response
        data = {
            "page": page,
            "items": [f"item-{page}-{i}" for i in range(3)],
            "has_more": page < max_pages
        }
        
        yield data
        
        if not data["has_more"]:
            break
        page += 1


async def generator_demo():
    """Demonstrate async generators."""
    
    # Basic async generator
    async for num in async_range(0, 5):
        print(f"Number: {num}")
    
    # Paginated API iteration
    async for page_data in fetch_pages("https://api.example.com", max_pages=3):
        print(f"Page {page_data['page']}: {page_data['items']}")


# ═══════════════════════════════════════════════════════════════
# ASYNC COMPREHENSIONS
# ═══════════════════════════════════════════════════════════════

async def comprehension_demo():
    """Demonstrate async comprehensions."""
    
    # Async list comprehension
    numbers = [num async for num in async_range(0, 5)]
    print(f"Numbers: {numbers}")
    
    # Async generator expression
    gen = (num * 2 async for num in async_range(0, 5))
    doubled = [n async for n in gen]
    print(f"Doubled: {doubled}")
    
    # With condition
    evens = [num async for num in async_range(0, 10) if num % 2 == 0]
    print(f"Evens: {evens}")
    
    # Async dict comprehension
    async def get_value(key):
        await asyncio.sleep(0.01)
        return key * 10
    
    keys = [1, 2, 3]
    mapping = {k: await get_value(k) for k in keys}
    print(f"Mapping: {mapping}")
```

## 3.5 Task Cancellation and Timeouts

```python
import asyncio
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# TASK CANCELLATION
# ═══════════════════════════════════════════════════════════════

async def cancellable_task(name: str):
    """Task that can be cancelled."""
    try:
        print(f"[{name}] Starting...")
        while True:
            await asyncio.sleep(1)
            print(f"[{name}] Working...")
    except asyncio.CancelledError:
        print(f"[{name}] Cancelled! Cleaning up...")
        # Perform cleanup
        await asyncio.sleep(0.1)  # Async cleanup is OK
        print(f"[{name}] Cleanup done")
        raise  # Re-raise to propagate cancellation


async def cancellation_demo():
    """Demonstrate task cancellation."""
    
    task = asyncio.create_task(cancellable_task("Worker"))
    
    await asyncio.sleep(2.5)
    
    print("Requesting cancellation...")
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")
    
    print(f"Task cancelled: {task.cancelled()}")


# ═══════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN PATTERN
# ═══════════════════════════════════════════════════════════════

class GracefulService:
    """Service with graceful shutdown."""
    
    def __init__(self):
        self.running = False
        self.tasks: list = []
    
    async def worker(self, worker_id: int):
        """Worker that handles cancellation gracefully."""
        try:
            while self.running:
                print(f"[Worker-{worker_id}] Processing...")
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print(f"[Worker-{worker_id}] Shutting down gracefully...")
            raise
        finally:
            print(f"[Worker-{worker_id}] Cleaned up")
    
    async def start(self, num_workers: int = 3):
        """Start the service."""
        self.running = True
        self.tasks = [
            asyncio.create_task(self.worker(i))
            for i in range(num_workers)
        ]
        print(f"Started {num_workers} workers")
    
    async def stop(self, timeout: float = 5.0):
        """Stop the service gracefully."""
        print("Stopping service...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for all to complete with timeout
        results = await asyncio.gather(
            *self.tasks,
            return_exceptions=True
        )
        
        print("Service stopped")
        return results


# ═══════════════════════════════════════════════════════════════
# TIMEOUT PATTERNS
# ═══════════════════════════════════════════════════════════════

async def timeout_patterns():
    """Various timeout patterns."""
    
    async def slow_operation():
        await asyncio.sleep(10)
        return "Done"
    
    # Pattern 1: wait_for with timeout
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out (wait_for)")
    
    # Pattern 2: timeout context manager (Python 3.11+)
    try:
        async with asyncio.timeout(2.0):
            result = await slow_operation()
    except TimeoutError:
        print("Operation timed out (context manager)")
    
    # Pattern 3: timeout with deadline
    try:
        deadline = asyncio.get_event_loop().time() + 2.0
        async with asyncio.timeout_at(deadline):
            result = await slow_operation()
    except TimeoutError:
        print("Operation timed out (deadline)")
    
    # Pattern 4: Manual timeout with shield
    task = asyncio.create_task(slow_operation())
    try:
        # shield() prevents cancellation of the inner task
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Timeout, but task continues in background")
        # task is still running!


# ═══════════════════════════════════════════════════════════════
# SHIELDING FROM CANCELLATION
# ═══════════════════════════════════════════════════════════════

async def critical_operation():
    """Operation that shouldn't be interrupted."""
    print("Starting critical operation...")
    await asyncio.sleep(2)
    print("Critical operation complete")
    return "Critical result"


async def shield_demo():
    """Demonstrate asyncio.shield."""
    
    async def wrapper():
        # Shield protects the inner coroutine from cancellation
        result = await asyncio.shield(critical_operation())
        return result
    
    task = asyncio.create_task(wrapper())
    
    await asyncio.sleep(0.5)
    task.cancel()
    
    try:
        result = await task
    except asyncio.CancelledError:
        print("Wrapper was cancelled, but critical operation continues")
        # Wait for the shielded operation to complete
        await asyncio.sleep(2)
```

## 3.6 Asyncio Pros and Cons

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ASYNCIO PROS & CONS                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✅ ADVANTAGES                                                               ║
║  ─────────────                                                               ║
║                                                                              ║
║  1. HIGH CONCURRENCY                                                         ║
║     • Handle 10,000+ simultaneous connections                                ║
║     • Tasks are lightweight (~2KB each vs ~8MB per thread)                   ║
║     • Minimal context-switching overhead                                     ║
║                                                                              ║
║  2. PREDICTABLE EXECUTION                                                    ║
║     • You control when context switches happen (at await)                    ║
║     • Easier to reason about shared state                                    ║
║     • No race conditions between await points                                ║
║                                                                              ║
║  3. SINGLE-THREADED                                                          ║
║     • No GIL issues                                                          ║
║     • Simpler mental model                                                   ║
║     • No need for thread synchronization primitives                          ║
║                                                                              ║
║  4. EXCELLENT FOR I/O-BOUND WORKLOADS                                        ║
║     • Network servers                                                        ║
║     • API clients                                                            ║
║     • Database operations                                                    ║
║     • File I/O                                                               ║
║                                                                              ║
║  5. STRUCTURED CONCURRENCY (3.11+)                                           ║
║     • TaskGroups provide clean error handling                                ║
║     • Automatic cleanup on exceptions                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ❌ DISADVANTAGES                                                            ║
║  ──────────────                                                              ║
║                                                                              ║
║  1. VIRAL NATURE                                                             ║
║     • async/await spreads through entire codebase                            ║
║     • Can't easily mix sync and async code                                   ║
║     • "Function coloring" problem                                            ║
║                                                                              ║
║  2. LEARNING CURVE                                                           ║
║     • New programming paradigm                                               ║
║     • Must understand event loop, coroutines, tasks                          ║
║     • Error messages can be confusing                                        ║
║                                                                              ║
║  3. ECOSYSTEM FRAGMENTATION                                                  ║
║     • Need async-compatible libraries                                        ║
║     • requests → aiohttp                                                     ║
║     • psycopg2 → asyncpg                                                     ║
║     • Not all libraries have async versions                                  ║
║                                                                              ║
║  4. DEBUGGING CHALLENGES                                                     ║
║     • Stack traces can be confusing                                          ║
║     • Harder to use traditional debuggers                                    ║
║     • Tricky to profile                                                      ║
║                                                                              ║
║  5. BLOCKING CODE DISASTER                                                   ║
║     • ONE blocking call blocks ENTIRE event loop                             ║
║     • Easy to accidentally use blocking libraries                            ║
║     • Must be vigilant about what code you call                              ║
║                                                                              ║
║  6. NOT FOR CPU-BOUND                                                        ║
║     • Still single-threaded                                                  ║
║     • CPU-heavy code blocks the loop                                         ║
║     • Must offload to thread/process pool                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

# 4. I/O Blocking: The Complete Reference

## 4.1 What is Blocking I/O?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BLOCKING vs NON-BLOCKING I/O                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BLOCKING I/O                                                               │
│  ─────────────                                                              │
│                                                                             │
│  Thread: ════════╗                    ╔══════════                           │
│                  ║    BLOCKED         ║                                     │
│                  ║    (waiting)       ║                                     │
│                  ╚════════════════════╝                                     │
│                     │                │                                      │
│                     ▼                ▼                                      │
│              Start I/O         I/O Complete                                 │
│                                                                             │
│  • Thread is SUSPENDED until I/O completes                                  │
│  • Cannot do any other work                                                 │
│  • OS puts thread to sleep                                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NON-BLOCKING I/O (asyncio)                                                 │
│  ──────────────────────────                                                 │
│                                                                             │
│  Task A: ════╗    ════════════╗    ════                                     │
│              ║                ║                                             │
│  Task B:     ╠════╗       ════╬════╗                                        │
│              ║    ║           ║    ║                                        │
│  Task C:     ║    ╠════╗      ║    ╠════                                    │
│              ║    ║    ║      ║    ║                                        │
│              ▼    ▼    ▼      ▼    ▼                                        │
│  Loop:  ═════════════════════════════════                                   │
│                                                                             │
│  • Tasks YIELD control at I/O points                                        │
│  • Event loop runs other tasks while waiting                                │
│  • Single thread serves multiple tasks                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4.2 Complete Blocking Operations Reference

```python
"""
═══════════════════════════════════════════════════════════════════════════════
                        COMPLETE BLOCKING OPERATIONS REFERENCE
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# NETWORK I/O - BLOCKING
# ═══════════════════════════════════════════════════════════════

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ LIBRARY              │ BLOCKING          │ ASYNC ALTERNATIVE               │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ requests             │ ✗ BLOCKING        │ aiohttp, httpx                  │
│ urllib, urllib3      │ ✗ BLOCKING        │ aiohttp, httpx                  │
│ socket               │ ✗ BLOCKING        │ asyncio streams                 │
│ http.client          │ ✗ BLOCKING        │ aiohttp                         │
│ ftplib               │ ✗ BLOCKING        │ aioftp                          │
│ smtplib              │ ✗ BLOCKING        │ aiosmtplib                      │
│ paramiko (SSH)       │ ✗ BLOCKING        │ asyncssh                        │
│ websocket-client     │ ✗ BLOCKING        │ websockets                      │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""

# Example: blocking vs non-blocking HTTP
import requests  # BLOCKING
import aiohttp   # NON-BLOCKING
import httpx     # SUPPORTS BOTH

# BLOCKING - freezes thread/loop
def blocking_fetch(url):
    response = requests.get(url)  # ← BLOCKS HERE
    return response.json()

# NON-BLOCKING - yields control while waiting
async def async_fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:  # ← YIELDS HERE
            return await response.json()


# ═══════════════════════════════════════════════════════════════
# DATABASE I/O - BLOCKING
# ═══════════════════════════════════════════════════════════════

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ LIBRARY              │ BLOCKING          │ ASYNC ALTERNATIVE               │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ psycopg2 (Postgres)  │ ✗ BLOCKING        │ asyncpg, psycopg3               │
│ PyMySQL              │ ✗ BLOCKING        │ aiomysql                        │
│ mysql-connector      │ ✗ BLOCKING        │ aiomysql                        │
│ sqlite3              │ ✗ BLOCKING        │ aiosqlite                       │
│ pymongo              │ ✗ BLOCKING        │ motor                           │
│ redis-py             │ ✗ BLOCKING        │ aioredis, redis.asyncio         │
│ SQLAlchemy (core)    │ ✗ BLOCKING        │ SQLAlchemy 1.4+ async           │
│ Django ORM           │ ✗ BLOCKING        │ Django 4.1+ async               │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""

# BLOCKING database
import psycopg2
def blocking_query():
    conn = psycopg2.connect(...)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")  # ← BLOCKS
    return cursor.fetchall()

# NON-BLOCKING database
import asyncpg
async def async_query():
    conn = await asyncpg.connect(...)
    rows = await conn.fetch("SELECT * FROM users")  # ← YIELDS
    return rows


# ═══════════════════════════════════════════════════════════════
# FILE I/O - BLOCKING
# ═══════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ OPERATION            │ BLOCKING          │ ASYNC ALTERNATIVE               │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ open(), read()       │ ✗ BLOCKING        │ aiofiles                        │
│ write()              │ ✗ BLOCKING        │ aiofiles                        │
│ os.listdir()         │ ✗ BLOCKING        │ aiofiles.os                     │
│ shutil operations    │ ✗ BLOCKING        │ asyncio.to_thread()             │
│ pathlib operations   │ ✗ BLOCKING        │ aiopath                         │
│ json.load()          │ ✗ BLOCKING        │ aiofiles + json.loads()         │
│ csv module           │ ✗ BLOCKING        │ aiocsv                          │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""

# BLOCKING file I/O
def blocking_read(filepath):
    with open(filepath, 'r') as f:
        return f.read()  # ← BLOCKS (especially for large files)

# NON-BLOCKING file I/O
import aiofiles
async def async_read(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        return await f.read()  # ← YIELDS


# ═══════════════════════════════════════════════════════════════
# SUBPROCESS / SYSTEM CALLS - BLOCKING
# ═══════════════════════════════════════════════════════════════

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ OPERATION            │ BLOCKING          │ ASYNC ALTERNATIVE               │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ subprocess.run()     │ ✗ BLOCKING        │ asyncio.create_subprocess_exec()│
│ subprocess.Popen()   │ ✗ wait() BLOCKS   │ asyncio.create_subprocess_shell │
│ os.system()          │ ✗ BLOCKING        │ asyncio subprocess              │
│ commands module      │ ✗ BLOCKING        │ asyncio subprocess              │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""

# BLOCKING subprocess
import subprocess
def blocking_command():
    result = subprocess.run(['ls', '-la'], capture_output=True)  # ← BLOCKS
    return result.stdout

# NON-BLOCKING subprocess
async def async_command():
    proc = await asyncio.create_subprocess_exec(
        'ls', '-la',
        stdout=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()  # ← YIELDS
    return stdout


# ═══════════════════════════════════════════════════════════════
# SLEEPING / WAITING - BLOCKING
# ═══════════════════════════════════════════════════════════════

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ OPERATION            │ BLOCKING          │ ASYNC ALTERNATIVE               │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ time.sleep()         │ ✗ BLOCKING        │ asyncio.sleep()                 │
│ threading.Event.wait │ ✗ BLOCKING        │ asyncio.Event.wait()            │
│ queue.Queue.get()    │ ✗ BLOCKING        │ asyncio.Queue.get()             │
│ Lock.acquire()       │ ✗ BLOCKING        │ asyncio.Lock.acquire()          │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""

# BLOCKING sleep - FREEZES THE EVENT LOOP!
import time
async def bad_async_function():
    time.sleep(5)  # ← BLOCKS ENTIRE EVENT LOOP! NO TASKS RUN!
    
# NON-BLOCKING sleep
async def good_async_function():
    await asyncio.sleep(5)  # ← YIELDS - other tasks can run


# ═══════════════════════════════════════════════════════════════
# CPU-BOUND OPERATIONS (Always "Blocking")
# ═══════════════════════════════════════════════════════════════

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ OPERATION                              │ SOLUTION                           │
├────────────────────────────────────────┼────────────────────────────────────┤
│ Mathematical computation               │ asyncio.to_thread() or             │
│ Data processing                        │ loop.run_in_executor()             │
│ Image processing                       │ ProcessPoolExecutor                │
│ Encryption/Decryption                  │                                    │
│ Compression/Decompression              │ For TRUE parallelism:              │
│ JSON parsing (large files)             │ multiprocessing                    │
│ Regular expressions (complex)          │                                    │
│ Machine learning inference             │                                    │
└────────────────────────────────────────┴────────────────────────────────────┘
"""

# CPU-bound in async context - WRONG
async def bad_cpu_work():
    result = heavy_computation()  # ← BLOCKS LOOP!
    return result

# CPU-bound in async context - CORRECT
async def good_cpu_work():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation)  # ← Thread pool
    return result

# OR with Python 3.9+
async def better_cpu_work():
    result = await asyncio.to_thread(heavy_computation)
    return result


# ═══════════════════════════════════════════════════════════════
# THIRD-PARTY LIBRARIES - COMMON GOTCHAS
# ═══════════════════════════════════════════════════════════════

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ LIBRARY              │ BLOCKING?         │ NOTES                           │
├──────────────────────┼───────────────────┼─────────────────────────────────┤
│ boto3 (AWS)          │ ✗ BLOCKING        │ Use aioboto3                    │
│ google-cloud         │ ✗ BLOCKING        │ Use gcloud-aio-*                │
│ azure-sdk            │ ✓ HAS ASYNC       │ Use azure-* aio versions        │
│ stripe               │ ✗ BLOCKING        │ Use asyncio.to_thread()         │
│ twilio               │ ✗ BLOCKING        │ Use asyncio.to_thread()         │
│ PIL/Pillow           │ ✗ BLOCKING        │ Use asyncio.to_thread()         │
│ OpenCV               │ ✗ BLOCKING        │ Use asyncio.to_thread()         │
│ BeautifulSoup        │ ✗ BLOCKING (CPU)  │ Use asyncio.to_thread()         │
│ lxml                 │ ✗ BLOCKING (CPU)  │ Use asyncio.to_thread()         │
│ pandas               │ ✗ BLOCKING (CPU)  │ Use asyncio.to_thread()         │
│ numpy                │ ✓ RELEASES GIL    │ Mostly OK, but heavy ops→thread │
└──────────────────────┴───────────────────┴─────────────────────────────────┘
"""
```

## 4.3 Identifying Blocking Code

```python
import asyncio
import time
import functools
from typing import Callable, Any

# ═══════════════════════════════════════════════════════════════
# DETECT SLOW CALLBACKS (Built-in debug mode)
# ═══════════════════════════════════════════════════════════════

# Run with debug mode to detect blocking
async def main():
    # This will trigger a warning in debug mode
    time.sleep(0.2)  # Blocking call!

# Enable debug mode
asyncio.run(main(), debug=True)
# Warning: Executing <Task...> took 0.200 seconds


# ═══════════════════════════════════════════════════════════════
# CUSTOM BLOCKING DETECTION DECORATOR
# ═══════════════════════════════════════════════════════════════

def detect_blocking(threshold: float = 0.1):
    """Decorator to detect blocking operations in async functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            start = loop.time()
            
            result = await func(*args, **kwargs)
            
            elapsed = loop.time() - start
            
            # Check if function took too long without yielding
            if elapsed > threshold:
                import warnings
                warnings.warn(
                    f"{func.__name__} took {elapsed:.3f}s - possible blocking call",
                    RuntimeWarning
                )
            
            return result
        return wrapper
    return decorator


@detect_blocking(threshold=0.05)
async def potentially_blocking():
    time.sleep(0.1)  # This will trigger warning
    return "done"


# ═══════════════════════════════════════════════════════════════
# BLOCKING DETECTION CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════

class BlockingDetector:
    """Context manager to detect blocking operations."""
    
    def __init__(self, threshold: float = 0.1, name: str = "operation"):
        self.threshold = threshold
        self.name = name
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = asyncio.get_event_loop().time() - self.start_time
        if elapsed > self.threshold:
            print(f"⚠️  BLOCKING DETECTED in '{self.name}': {elapsed:.3f}s")
        return False


async def example_with_detector():
    async with BlockingDetector(threshold=0.05, name="my_operation"):
        time.sleep(0.1)  # Will be detected!


# ═══════════════════════════════════════════════════════════════
# MONITORING EVENT LOOP LAG
# ═══════════════════════════════════════════════════════════════

class EventLoopMonitor:
    """Monitor event loop for blocking."""
    
    def __init__(self, interval: float = 0.1, threshold: float = 0.15):
        self.interval = interval
        self.threshold = threshold
        self._running = False
        self._task = None
    
    async def start(self):
        """Start monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor())
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _monitor(self):
        """Monitor loop for lag."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            start = loop.time()
            await asyncio.sleep(self.interval)
            elapsed = loop.time() - start
            
            lag = elapsed - self.interval
            if lag > self.threshold - self.interval:
                print(f"⚠️  EVENT LOOP LAG: {lag*1000:.1f}ms")


# Usage
async def monitored_main():
    monitor = EventLoopMonitor()
    await monitor.start()
    
    # Your async code here...
    await asyncio.sleep(1)
    
    # This will cause lag
    time.sleep(0.3)
    
    await asyncio.sleep(0.5)
    await monitor.stop()
```

---

# 5. Comprehensive Comparison

## 5.1 Side-by-Side Feature Comparison

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    THREADING vs ASYNCIO - COMPLETE COMPARISON                        ║
╠═══════════════════════════╦══════════════════════════╦═══════════════════════════════╣
║ FEATURE                   ║ THREADING                ║ ASYNCIO                       ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Concurrency Type          ║ Preemptive               ║ Cooperative                   ║
║                           ║ (OS controlled)          ║ (Application controlled)      ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Parallelism               ║ No (GIL)                 ║ No (single-threaded)          ║
║                           ║ Yes for I/O ops          ║                               ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Memory per unit           ║ ~1-8 MB per thread       ║ ~2-3 KB per task              ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Max concurrent units      ║ Hundreds to low          ║ Tens of thousands             ║
║                           ║ thousands                ║                               ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Context switch cost       ║ High (OS level)          ║ Low (Python level)            ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Race conditions           ║ High risk                ║ Lower risk (only at           ║
║                           ║ (can happen anywhere)    ║ await points)                 ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Deadlocks possible        ║ Yes                      ║ Yes (but less common)         ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Debugging                 ║ Difficult                ║ Easier (deterministic)        ║
║                           ║ (non-deterministic)      ║                               ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Library compatibility     ║ Universal                ║ Requires async libraries      ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Learning curve            ║ Lower                    ║ Higher                        ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Code changes needed       ║ Minimal                  ║ Significant (async/await)     ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ I/O bound efficiency      ║ Good                     ║ Excellent                     ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ CPU bound efficiency      ║ Poor (GIL)               ║ Poor (single thread)          ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Blocking code handling    ║ Native support           ║ Needs run_in_executor         ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Shared state              ║ Requires locks           ║ Generally safer               ║
╠═══════════════════════════╬══════════════════════════╬═══════════════════════════════╣
║ Error handling            ║ Standard try/except      ║ More complex (futures)        ║
╚═══════════════════════════╩══════════════════════════╩═══════════════════════════════╝
```

## 5.2 Performance Comparison

```python
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

# ═══════════════════════════════════════════════════════════════
# BENCHMARK: I/O-BOUND TASKS
# ═══════════════════════════════════════════════════════════════

def io_bound_sync(duration: float):
    """Simulate I/O with sleep."""
    time.sleep(duration)
    return duration

async def io_bound_async(duration: float):
    """Simulate I/O with async sleep."""
    await asyncio.sleep(duration)
    return duration

def benchmark_threading(num_tasks: int, duration: float) -> float:
    """Benchmark threading approach."""
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(io_bound_sync, duration) for _ in range(num_tasks)]
        results = [f.result() for f in futures]
    
    return time.perf_counter() - start

async def benchmark_asyncio(num_tasks: int, duration: float) -> float:
    """Benchmark asyncio approach."""
    start = time.perf_counter()
    
    tasks = [io_bound_async(duration) for _ in range(num_tasks)]
    results = await asyncio.gather(*tasks)
    
    return time.perf_counter() - start

def run_benchmarks():
    """Run and compare benchmarks."""
    
    test_cases = [
        (10, 0.1),    # 10 tasks, 0.1s each
        (100, 0.1),   # 100 tasks, 0.1s each
        (1000, 0.1),  # 1000 tasks, 0.1s each
    ]
    
    print("="*70)
    print("I/O-BOUND BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Tasks':<10} {'Duration':<10} {'Threading':<15} {'Asyncio':<15} {'Winner':<10}")
    print("-"*70)
    
    for num_tasks, duration in test_cases:
        threading_time = benchmark_threading(num_tasks, duration)
        asyncio_time = asyncio.run(benchmark_asyncio(num_tasks, duration))
        
        winner = "Asyncio" if asyncio_time < threading_time else "Threading"
        
        print(f"{num_tasks:<10} {duration:<10} {threading_time:<15.3f} {asyncio_time:<15.3f} {winner:<10}")

# run_benchmarks()


# ═══════════════════════════════════════════════════════════════
# MEMORY COMPARISON
# ═══════════════════════════════════════════════════════════════

"""
Memory Usage Comparison (Approximate):

┌────────────────────────────────────────────────────────────────────────────┐
│ Concurrent Units    │ Threading Memory      │ Asyncio Memory               │
├─────────────────────┼───────────────────────┼──────────────────────────────┤
│ 100                 │ ~100-800 MB           │ ~0.2-0.3 MB                  │
│ 1,000               │ ~1-8 GB               │ ~2-3 MB                      │
│ 10,000              │ ~10-80 GB (!)         │ ~20-30 MB                    │
│ 100,000             │ Not practical         │ ~200-300 MB                  │
└─────────────────────┴───────────────────────┴──────────────────────────────┘

Note: Thread stack size varies by OS. Default is often 1-8 MB per thread.
      Asyncio tasks are just Python objects (~2-3 KB each).
"""
```

---

## 5.3 Decision Flowchart

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Is your workload     │
                    │    CPU-bound?         │
                    └───────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │YES                              │NO
              ▼                                 ▼
    ┌─────────────────────┐         ┌───────────────────────┐
    │   Use multiprocess  │         │  Is your workload     │
    │   ProcessPoolExec.  │         │    I/O-bound?         │
    └─────────────────────┘         └───────────┬───────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │YES                              │NO
                              ▼                                 ▼
                  ┌───────────────────────┐         ┌──────────────────────┐
                  │ Do you need 1000+     │         │  Probably don't need │
                  │ concurrent operations?│         │  concurrency         │
                  └───────────┬───────────┘         └──────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │YES                              │NO
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│     Use ASYNCIO       │         │ Can you use async     │
│                       │         │ libraries?            │
│ • High concurrency    │         └───────────┬───────────┘
│ • Low memory usage    │                    │
│ • WebSockets, HTTP    │       ┌────────────┴────────────┐
└───────────────────────┘       │YES                      │NO
                                ▼                         ▼
                    ┌───────────────────┐     ┌───────────────────────┐
                    │   Use ASYNCIO     │     │   Use THREADING       │
                    │                   │     │                       │
                    │ • Modern approach │     │ • Legacy code         │
                    │ • Better perf     │     │ • Blocking libraries  │
                    │ • Cleaner code    │     │ • Simple background   │
                    └───────────────────┘     │   tasks               │
                                              └───────────────────────┘
```

## 5.4 Decision Matrix

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                           WHEN TO USE WHAT - QUICK REFERENCE                         ║
╠════════════════════════════════════════════╦═════════════════════════════════════════╣
║              SCENARIO                      ║           RECOMMENDED APPROACH          ║
╠════════════════════════════════════════════╬═════════════════════════════════════════╣
║ Web scraping (few URLs)                    ║ Threading + requests                    ║
║ Web scraping (many URLs)                   ║ Asyncio + aiohttp                       ║
║ HTTP API server                            ║ Asyncio (FastAPI, aiohttp)              ║
║ WebSocket server                           ║ Asyncio (websockets)                    ║
║ Database migrations                        ║ Synchronous (sequential)                ║
║ Bulk database operations                   ║ Asyncio + asyncpg/aiomysql              ║
║ File processing (few files)                ║ Threading                               ║
║ File processing (many files)               ║ Asyncio + aiofiles                      ║
║ Image processing                           ║ Multiprocessing                         ║
║ ML model training                          ║ Multiprocessing / GPU frameworks        ║
║ ML model inference (API)                   ║ Asyncio + thread pool for inference     ║
║ Chat application                           ║ Asyncio                                 ║
║ GUI application                            ║ Threading (for background tasks)        ║
║ CLI tool with progress                     ║ Threading or Asyncio                    ║
║ Microservices                              ║ Asyncio                                 ║
║ Legacy system integration                  ║ Threading                               ║
║ Real-time data streaming                   ║ Asyncio                                 ║
║ Scheduled background jobs                  ║ Threading (or Celery for distributed)   ║
╚════════════════════════════════════════════╩═════════════════════════════════════════╝
```

---

# 6. Combining Threading and Asyncio

## 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE: ASYNCIO + THREADING                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         MAIN ASYNC EVENT LOOP                               │    │
│  │                         (Single Thread)                                     │    │
│  │                                                                             │    │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │    │
│  │   │  Task 1  │  │  Task 2  │  │  Task 3  │  │  Task N  │                    │    │
│  │   │  (async) │  │  (async) │  │  (async) │  │  (async) │                    │    │
│  │   └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘                    │    │
│  │        │             │                                                      │    │
│  │        │             │ Blocking call needed                                 │    │
│  │        │             ▼                                                      │    │
│  │        │    ┌─────────────────────┐                                         │    │
│  │        │    │ asyncio.to_thread() │                                         │    │
│  │        │    │        or           │                                         │    │
│  │        │    │ run_in_executor()   │                                         │    │
│  │        │    └──────────┬──────────┘                                         │    │
│  └────────┼───────────────┼────────────────────────────────────────────────────┘    │
│           │               │                                                         │
│           │               ▼                                                         │
│  ┌────────┼─────────────────────────────────────────────────────────────────────┐   │
│  │        │         THREAD POOL (ThreadPoolExecutor)                            │   │
│  │        │                                                                     │   │
│  │        │    ┌──────────┐  ┌──────────┐  ┌──────────┐                         │   │
│  │        │    │ Worker 1 │  │ Worker 2 │  │ Worker 3 │                         │   │
│  │        │    │(blocking)│  │(blocking)│  │(blocking)│                         │   │
│  │        │    │  code    │  │  code    │  │  code    │                         │   │
│  │        │    └──────────┘  └──────────┘  └──────────┘                         │   │
│  │        │                                                                     │   │
│  └────────┼─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                         │
│           ▼                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                    PROCESS POOL (for CPU-bound)                              │   │
│  │                                                                              │   │
│  │    ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐                │   │
│  │    │ Process 1 │  │ Process 2 │  │ Process 3 │  │ Process 4 │                │   │
│  │    │ (CPU work)│  │ (CPU work)│  │ (CPU work)│  │ (CPU work)│                │   │
│  │    └───────────┘  └───────────┘  └───────────┘  └───────────┘                │   │
│  │                                                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 6.2 Running Blocking Code in Async Context

```python
import asyncio
import time
import concurrent.futures
from functools import partial
from typing import Any, Callable

# ═══════════════════════════════════════════════════════════════
# METHOD 1: asyncio.to_thread() - Python 3.9+ (Recommended)
# ═══════════════════════════════════════════════════════════════

def blocking_io_operation(url: str) -> dict:
    """Simulates a blocking I/O operation."""
    import requests  # Blocking library
    # response = requests.get(url)
    # return response.json()
    time.sleep(1)  # Simulate blocking
    return {"url": url, "status": "ok"}


def cpu_intensive_task(data: list) -> int:
    """CPU-bound task."""
    total = 0
    for i in range(10_000_000):
        total += i
    return total


async def using_to_thread():
    """Use asyncio.to_thread for blocking operations."""
    
    print("Starting async operations...")
    
    # Run multiple blocking operations concurrently
    results = await asyncio.gather(
        asyncio.to_thread(blocking_io_operation, "https://api1.example.com"),
        asyncio.to_thread(blocking_io_operation, "https://api2.example.com"),
        asyncio.to_thread(blocking_io_operation, "https://api3.example.com"),
    )
    
    print(f"Results: {results}")
    return results


# ═══════════════════════════════════════════════════════════════
# METHOD 2: loop.run_in_executor() - More control
# ═══════════════════════════════════════════════════════════════

async def using_run_in_executor():
    """Use run_in_executor for fine-grained control."""
    
    loop = asyncio.get_running_loop()
    
    # Default executor (ThreadPoolExecutor)
    result1 = await loop.run_in_executor(
        None,  # None = use default executor
        blocking_io_operation,
        "https://api.example.com"
    )
    
    # Custom thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as thread_pool:
        result2 = await loop.run_in_executor(
            thread_pool,
            blocking_io_operation,
            "https://api2.example.com"
        )
    
    # Process pool for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as process_pool:
        result3 = await loop.run_in_executor(
            process_pool,
            cpu_intensive_task,
            [1, 2, 3, 4, 5]
        )
    
    return result1, result2, result3


# ═══════════════════════════════════════════════════════════════
# METHOD 3: Using functools.partial for arguments
# ═══════════════════════════════════════════════════════════════

def blocking_with_kwargs(url: str, timeout: int = 30, headers: dict = None) -> dict:
    """Blocking function with keyword arguments."""
    time.sleep(0.5)
    return {"url": url, "timeout": timeout, "headers": headers}


async def using_partial():
    """Use partial for functions with keyword arguments."""
    
    loop = asyncio.get_running_loop()
    
    # Create partial function with kwargs
    func = partial(
        blocking_with_kwargs,
        timeout=60,
        headers={"Authorization": "Bearer token"}
    )
    
    result = await loop.run_in_executor(None, func, "https://api.example.com")
    return result


# ═══════════════════════════════════════════════════════════════
# HELPER: Decorator to automatically run blocking code in thread
# ═══════════════════════════════════════════════════════════════

def run_in_thread(func: Callable) -> Callable:
    """Decorator to run blocking function in thread pool."""
    
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        # Use partial to handle kwargs
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, pfunc)
    
    return wrapper


@run_in_thread
def my_blocking_function(x: int, y: int) -> int:
    """This blocking function will automatically run in a thread."""
    time.sleep(1)
    return x + y


async def using_decorator():
    """Use the decorator for cleaner code."""
    
    # Now can be awaited directly
    result = await my_blocking_function(10, 20)
    print(f"Result: {result}")


# ═══════════════════════════════════════════════════════════════
# COMPLETE EXAMPLE: Mixing async and blocking code
# ═══════════════════════════════════════════════════════════════

class HybridService:
    """Service that combines async and blocking operations."""
    
    def __init__(self, max_workers: int = 10):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    
    async def fetch_async(self, url: str) -> dict:
        """Native async HTTP call."""
        # Using aiohttp would be:
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url) as response:
        #         return await response.json()
        await asyncio.sleep(0.5)  # Simulate
        return {"url": url, "async": True}
    
    async def fetch_blocking(self, url: str) -> dict:
        """Blocking HTTP call wrapped for async."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            blocking_io_operation,
            url
        )
    
    async def process_cpu_bound(self, data: Any) -> Any:
        """CPU-bound processing in separate process."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool,
            cpu_intensive_task,
            data
        )
    
    async def complex_operation(self, urls: list, data: Any) -> dict:
        """Combine multiple operation types."""
        
        # Run all concurrently
        async_results, blocking_results, cpu_result = await asyncio.gather(
            # Async operations
            asyncio.gather(*[self.fetch_async(url) for url in urls[:2]]),
            # Blocking operations in threads
            asyncio.gather(*[self.fetch_blocking(url) for url in urls[2:]]),
            # CPU-bound in process
            self.process_cpu_bound(data),
        )
        
        return {
            "async_results": async_results,
            "blocking_results": blocking_results,
            "cpu_result": cpu_result,
        }
    
    def shutdown(self):
        """Clean up executors."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


async def main():
    service = HybridService()
    try:
        urls = [f"https://api{i}.example.com" for i in range(5)]
        result = await service.complex_operation(urls, [1, 2, 3])
        print(result)
    finally:
        service.shutdown()


# asyncio.run(main())
```

## 6.3 Running Async Code from Sync/Threading Context

```python
import asyncio
import threading
import time
from typing import Any, Coroutine
from concurrent.futures import Future

# ═══════════════════════════════════════════════════════════════
# SCENARIO: You have a sync application and need to call async code
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# Pattern 1: Simple one-off async call from sync code
# ─────────────────────────────────────────────────────────────────

async def async_operation(value: int) -> int:
    """Some async operation."""
    await asyncio.sleep(1)
    return value * 2


def sync_function_simple():
    """Call async code from sync context - simple case."""
    # This creates a new event loop, runs the coroutine, and closes the loop
    result = asyncio.run(async_operation(21))
    print(f"Result: {result}")
    return result


# ─────────────────────────────────────────────────────────────────
# Pattern 2: Persistent event loop in background thread
# ─────────────────────────────────────────────────────────────────

class AsyncRunner:
    """
    Runs an event loop in a background thread.
    Allows sync code to submit async work.
    """
    
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._started = threading.Event()
    
    def start(self):
        """Start the background event loop."""
        if self._thread is not None:
            return
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait()  # Wait for loop to be ready
    
    def _run_loop(self):
        """Run event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()
    
    def run(self, coro: Coroutine) -> Any:
        """
        Run a coroutine from sync code and wait for result.
        Thread-safe.
        """
        if self._loop is None:
            raise RuntimeError("AsyncRunner not started")
        
        # Submit coroutine to the loop thread
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        # Wait for result (blocks current thread)
        return future.result()
    
    def run_nowait(self, coro: Coroutine) -> Future:
        """
        Run a coroutine without waiting.
        Returns a concurrent.futures.Future.
        """
        if self._loop is None:
            raise RuntimeError("AsyncRunner not started")
        
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def stop(self):
        """Stop the background event loop."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop = None
            self._thread = None


# Usage example
def sync_application():
    """Sync application using AsyncRunner."""
    
    runner = AsyncRunner()
    runner.start()
    
    try:
        # Synchronous call that waits for result
        result = runner.run(async_operation(42))
        print(f"Sync got result: {result}")
        
        # Fire-and-forget
        future = runner.run_nowait(async_operation(100))
        
        # Do other sync work...
        time.sleep(0.5)
        
        # Check result later
        if future.done():
            print(f"Async result ready: {future.result()}")
        else:
            print("Still waiting...")
            result = future.result(timeout=5)
            print(f"Got result: {result}")
    
    finally:
        runner.stop()


# ─────────────────────────────────────────────────────────────────
# Pattern 3: Thread-safe bridge for async/sync communication
# ─────────────────────────────────────────────────────────────────

class AsyncSyncBridge:
    """
    Bridge between async and sync worlds.
    Useful for frameworks like Flask/Django with async components.
    """
    
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._ready = threading.Event()
    
    def start(self):
        """Start background async loop."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
    
    def _run(self):
        """Background thread running event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
    
    def call_async(self, coro: Coroutine, timeout: float = None) -> Any:
        """Call async function from sync context."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
    
    def call_async_callback(self, coro: Coroutine, callback: callable):
        """Call async function with callback when done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(lambda f: callback(f.result()))
    
    def schedule(self, coro: Coroutine) -> Future:
        """Schedule coroutine without waiting."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def stop(self):
        """Stop the bridge."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────
# Pattern 4: Context manager for temporary async context
# ─────────────────────────────────────────────────────────────────

class AsyncContext:
    """Context manager providing async capabilities to sync code."""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self._ready = threading.Event()
    
    def __enter__(self):
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        self._ready.wait()
        return self
    
    def __exit__(self, *args):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
    
    def _start_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()
    
    def run(self, coro: Coroutine) -> Any:
        """Run coroutine and return result."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()


# Usage
def use_async_context():
    """Use async context manager."""
    
    with AsyncContext() as ctx:
        result1 = ctx.run(async_operation(10))
        result2 = ctx.run(async_operation(20))
        print(f"Results: {result1}, {result2}")
```

## 6.4 Thread-Safe Communication Between Async and Sync

```python
import asyncio
import threading
import queue
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# Using janus - Dual-interface queue (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════

# pip install janus

import janus

async def async_consumer(async_queue: janus.AsyncQueue):
    """Async consumer reading from queue."""
    while True:
        item = await async_queue.get()
        if item is None:
            break
        print(f"[Async Consumer] Got: {item}")
        async_queue.task_done()


def sync_producer(sync_queue: janus.SyncQueue, items: list):
    """Sync producer writing to queue."""
    for item in items:
        sync_queue.put(item)
        print(f"[Sync Producer] Put: {item}")
    sync_queue.put(None)  # Sentinel


async def janus_example():
    """Demonstrate janus queue."""
    
    # Create dual-interface queue
    mixed_queue = janus.Queue()
    
    # Start async consumer
    consumer_task = asyncio.create_task(async_consumer(mixed_queue.async_q))
    
    # Run sync producer in thread
    await asyncio.to_thread(sync_producer, mixed_queue.sync_q, [1, 2, 3, 4, 5])
    
    # Wait for consumer to finish
    await consumer_task
    
    # Cleanup
    mixed_queue.close()
    await mixed_queue.wait_closed()


# ═══════════════════════════════════════════════════════════════
# Manual implementation without janus
# ═══════════════════════════════════════════════════════════════

class AsyncSyncQueue:
    """
    Thread-safe queue that works with both async and sync code.
    Custom implementation without external dependencies.
    """
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._async_event = None
        self._loop = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop reference."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
        return self._loop
    
    def _notify_async(self):
        """Notify async waiters that item is available."""
        if self._async_event and self._loop:
            self._loop.call_soon_threadsafe(self._async_event.set)
    
    # Sync interface
    def put_sync(self, item: Any, block: bool = True, timeout: float = None):
        """Put item from sync context."""
        self._queue.put(item, block=block, timeout=timeout)
        self._notify_async()
    
    def get_sync(self, block: bool = True, timeout: float = None) -> Any:
        """Get item from sync context."""
        return self._queue.get(block=block, timeout=timeout)
    
    # Async interface
    async def put_async(self, item: Any):
        """Put item from async context."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._queue.put, item)
    
    async def get_async(self) -> Any:
        """Get item from async context."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        
        while True:
            try:
                # Try non-blocking get first
                return self._queue.get_nowait()
            except queue.Empty:
                # Wait for notification
                self._async_event = asyncio.Event()
                await self._async_event.wait()
                self._async_event.clear()
    
    def qsize(self) -> int:
        return self._queue.qsize()
    
    def empty(self) -> bool:
        return self._queue.empty()


# ═══════════════════════════════════════════════════════════════
# Message-based communication pattern
# ═══════════════════════════════════════════════════════════════

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Message for async-sync communication."""
    type: MessageType
    id: str
    payload: Any = None
    error: str = None


class MessageBus:
    """
    Message bus for async-sync communication.
    Supports request-response patterns.
    """
    
    def __init__(self):
        self._request_queue = janus.Queue()
        self._response_queues: dict = {}
        self._lock = threading.Lock()
    
    # Sync side (e.g., Flask routes)
    def send_request_sync(self, message_id: str, payload: Any, timeout: float = 30) -> Any:
        """Send request from sync context and wait for response."""
        
        # Create response queue for this request
        response_queue = queue.Queue()
        with self._lock:
            self._response_queues[message_id] = response_queue
        
        try:
            # Send request
            message = Message(
                type=MessageType.REQUEST,
                id=message_id,
                payload=payload
            )
            self._request_queue.sync_q.put(message)
            
            # Wait for response
            response: Message = response_queue.get(timeout=timeout)
            
            if response.type == MessageType.ERROR:
                raise Exception(response.error)
            
            return response.payload
        
        finally:
            with self._lock:
                del self._response_queues[message_id]
    
    # Async side (e.g., async service)
    async def receive_request_async(self) -> Message:
        """Receive request in async context."""
        return await self._request_queue.async_q.get()
    
    async def send_response_async(self, message_id: str, payload: Any = None, error: str = None):
        """Send response from async context."""
        
        with self._lock:
            response_queue = self._response_queues.get(message_id)
        
        if response_queue is None:
            return  # Request already timed out
        
        message = Message(
            type=MessageType.ERROR if error else MessageType.RESPONSE,
            id=message_id,
            payload=payload,
            error=error
        )
        
        response_queue.put(message)
    
    def close(self):
        """Close the message bus."""
        self._request_queue.close()


# ═══════════════════════════════════════════════════════════════
# Complete example: Web server with async backend
# ═══════════════════════════════════════════════════════════════

class AsyncBackend:
    """Async backend service."""
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self._running = False
    
    async def process_request(self, message: Message) -> Any:
        """Process a request."""
        # Simulate async processing
        await asyncio.sleep(0.1)
        return f"Processed: {message.payload}"
    
    async def run(self):
        """Main loop processing requests."""
        self._running = True
        
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.bus.receive_request_async(),
                    timeout=1.0
                )
                
                if message.type == MessageType.SHUTDOWN:
                    break
                
                try:
                    result = await self.process_request(message)
                    await self.bus.send_response_async(message.id, payload=result)
                except Exception as e:
                    await self.bus.send_response_async(message.id, error=str(e))
            
            except asyncio.TimeoutError:
                continue
    
    def stop(self):
        self._running = False


def sync_web_handler(message_bus: MessageBus, request_data: dict) -> dict:
    """Simulated sync web handler (e.g., Flask route)."""
    import uuid
    
    message_id = str(uuid.uuid4())
    
    try:
        result = message_bus.send_request_sync(message_id, request_data, timeout=10)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def run_complete_example():
    """Run the complete async-sync integration example."""
    
    bus = MessageBus()
    backend = AsyncBackend(bus)
    
    # Start async backend
    backend_task = asyncio.create_task(backend.run())
    
    # Simulate sync requests from threads
    def make_requests():
        for i in range(5):
            result = sync_web_handler(bus, {"request_id": i})
            print(f"Request {i} result: {result}")
    
    # Run sync code in thread
    await asyncio.to_thread(make_requests)
    
    # Shutdown
    backend.stop()
    await backend_task
    bus.close()


# asyncio.run(run_complete_example())
```

---

# 7. Real-World Patterns and Recipes

## 7.1 Web Scraper with Rate Limiting

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
import time

@dataclass
class ScrapedPage:
    url: str
    status: int
    content: str
    elapsed: float
    error: Optional[str] = None


class AsyncWebScraper:
    """
    Production-ready async web scraper with:
    - Rate limiting
    - Concurrent connection limits
    - Retry logic
    - Error handling
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        requests_per_second: float = 5.0,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._semaphore: asyncio.Semaphore = None
        self._rate_limiter: asyncio.Lock = None
        self._last_request_time: float = 0
        self._session: aiohttp.ClientSession = None
    
    async def __aenter__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._rate_limiter = asyncio.Lock()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, *args):
        await self._session.close()
    
    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limit."""
        async with self._rate_limiter:
            now = time.monotonic()
            min_interval = 1.0 / self.requests_per_second
            time_since_last = now - self._last_request_time
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            self._last_request_time = time.monotonic()
    
    async def fetch_one(self, url: str) -> ScrapedPage:
        """Fetch a single URL with retries."""
        
        async with self._semaphore:
            await self._wait_for_rate_limit()
            
            for attempt in range(self.max_retries):
                start = time.monotonic()
                
                try:
                    async with self._session.get(url) as response:
                        content = await response.text()
                        return ScrapedPage(
                            url=url,
                            status=response.status,
                            content=content,
                            elapsed=time.monotonic() - start
                        )
                
                except asyncio.TimeoutError:
                    error = f"Timeout (attempt {attempt + 1})"
                except aiohttp.ClientError as e:
                    error = f"{type(e).__name__}: {e} (attempt {attempt + 1})"
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            
            return ScrapedPage(
                url=url,
                status=0,
                content="",
                elapsed=time.monotonic() - start,
                error=error
            )
    
    async def fetch_many(self, urls: List[str]) -> List[ScrapedPage]:
        """Fetch multiple URLs concurrently."""
        tasks = [self.fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    async def fetch_with_progress(
        self, 
        urls: List[str], 
        callback: callable = None
    ) -> List[ScrapedPage]:
        """Fetch with progress callback."""
        
        results = []
        total = len(urls)
        
        for i, coro in enumerate(asyncio.as_completed(
            [self.fetch_one(url) for url in urls]
        )):
            result = await coro
            results.append(result)
            
            if callback:
                callback(i + 1, total, result)
        
        return results


async def scraper_example():
    """Example usage of the scraper."""
    
    urls = [f"https://httpbin.org/delay/{i % 3}" for i in range(10)]
    
    def progress(done, total, result):
        status = "✓" if result.status == 200 else "✗"
        print(f"[{done}/{total}] {status} {result.url} ({result.elapsed:.2f}s)")
    
    async with AsyncWebScraper(
        max_concurrent=5,
        requests_per_second=2.0
    ) as scraper:
        results = await scraper.fetch_with_progress(urls, progress)
    
    successful = sum(1 for r in results if r.status == 200)
    print(f"\nCompleted: {successful}/{len(results)} successful")


# asyncio.run(scraper_example())
```

## 7.2 Connection Pool with Health Checks

```python
import asyncio
from typing import Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

@dataclass
class Connection:
    """Represents a database/service connection."""
    id: int
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    use_count: int = 0
    
    async def execute(self, query: str) -> Any:
        """Execute a query on this connection."""
        self.last_used = datetime.now()
        self.use_count += 1
        await asyncio.sleep(0.1)  # Simulate query
        return f"Result from connection {self.id}: {query}"
    
    async def health_check(self) -> bool:
        """Check if connection is healthy."""
        await asyncio.sleep(0.05)  # Simulate health check
        self.is_healthy = random.random() > 0.1  # 90% healthy
        return self.is_healthy
    
    async def close(self):
        """Close the connection."""
        await asyncio.sleep(0.01)
        print(f"Connection {self.id} closed")


class AsyncConnectionPool:
    """
    Production-ready async connection pool with:
    - Min/max connections
    - Health checks
    - Connection recycling
    - Automatic scaling
    """
    
    def __init__(
        self,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: timedelta = timedelta(minutes=5),
        health_check_interval: timedelta = timedelta(seconds=30)
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        self._pool: asyncio.Queue = None
        self._all_connections: List[Connection] = []
        self._connection_counter = 0
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task = None
        self._maintenance_task: asyncio.Task = None
        self._closed = False
    
    async def start(self):
        """Initialize the pool."""
        self._pool = asyncio.Queue()
        self._closed = False
        
        # Create minimum connections
        for _ in range(self.min_size):
            conn = await self._create_connection()
            await self._pool.put(conn)
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        print(f"Pool started with {self.min_size} connections")
    
    async def _create_connection(self) -> Connection:
        """Create a new connection."""
        async with self._lock:
            self._connection_counter += 1
            conn = Connection(id=self._connection_counter)
            self._all_connections.append(conn)
            print(f"Created connection {conn.id}")
            return conn
    
    async def acquire(self, timeout: float = 10.0) -> Connection:
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")
        
        try:
            # Try to get from pool
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=timeout
            )
            
            # Check if connection is still healthy
            if not conn.is_healthy:
                await conn.close()
                self._all_connections.remove(conn)
                conn = await self._create_connection()
            
            return conn
        
        except asyncio.TimeoutError:
            # Pool exhausted - try to create new connection
            async with self._lock:
                if len(self._all_connections) < self.max_size:
                    return await self._create_connection()
            raise RuntimeError("Connection pool exhausted")
    
    async def release(self, conn: Connection):
        """Return a connection to the pool."""
        if self._closed:
            await conn.close()
            return
        
        conn.last_used = datetime.now()
        await self._pool.put(conn)
    
    async def _health_check_loop(self):
        """Periodically check connection health."""
        while not self._closed:
            await asyncio.sleep(self.health_check_interval.total_seconds())
            
            for conn in self._all_connections[:]:
                if not self._closed:
                    is_healthy = await conn.health_check()
                    if not is_healthy:
                        print(f"Connection {conn.id} marked unhealthy")
    
    async def _maintenance_loop(self):
        """Perform pool maintenance."""
        while not self._closed:
            await asyncio.sleep(60)  # Check every minute
            
            now = datetime.now()
            
            async with self._lock:
                # Remove idle connections above minimum
                idle_connections = [
                    c for c in self._all_connections
                    if (now - c.last_used) > self.max_idle_time
                ]
                
                for conn in idle_connections:
                    if len(self._all_connections) > self.min_size:
                        self._all_connections.remove(conn)
                        await conn.close()
                        print(f"Removed idle connection {conn.id}")
    
    async def close(self):
        """Close the pool."""
        self._closed = True
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._maintenance_task:
            self._maintenance_task.cancel()
        
        # Close all connections
        for conn in self._all_connections:
            await conn.close()
        
        self._all_connections.clear()
        print("Pool closed")
    
    @property
    def size(self) -> int:
        """Current pool size."""
        return len(self._all_connections)
    
    @property
    def available(self) -> int:
        """Available connections."""
        return self._pool.qsize() if self._pool else 0


# Context manager for acquiring connections
class PooledConnection:
    """Context manager for automatic connection release."""
    
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool
        self.connection: Connection = None
    
    async def __aenter__(self) -> Connection:
        self.connection = await self.pool.acquire()
        return self.connection
    
    async def __aexit__(self, *args):
        if self.connection:
            await self.pool.release(self.connection)


async def pool_example():
    """Example usage of connection pool."""
    
    pool = AsyncConnectionPool(min_size=2, max_size=5)
    await pool.start()
    
    try:
        # Use connection with context manager
        async with PooledConnection(pool) as conn:
            result = await conn.execute("SELECT * FROM users")
            print(result)
        
        # Multiple concurrent queries
        async def query(n: int):
            async with PooledConnection(pool) as conn:
                return await conn.execute(f"Query {n}")
        
        results = await asyncio.gather(*[query(i) for i in range(10)])
        for r in results:
            print(r)
    
    finally:
        await pool.close()


# asyncio.run(pool_example())
```

## 7.3 Producer-Consumer with Backpressure

```python
import asyncio
from typing import Generic, TypeVar, Callable, Awaitable, Optional
from dataclasses import dataclass
from datetime import datetime
import random

T = TypeVar('T')


@dataclass
class WorkItem(Generic[T]):
    """Work item with metadata."""
    id: int
    data: T
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BackpressureQueue(Generic[T]):
    """
    Queue with backpressure support.
    Slows down producers when consumers can't keep up.
    """
    
    def __init__(
        self,
        maxsize: int = 100,
        high_water_mark: float = 0.8,
        low_water_mark: float = 0.3
    ):
        self.maxsize = maxsize
        self.high_water_mark = high_water_mark
        self.low_water_mark = low_water_mark
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._backpressure_event = asyncio.Event()
        self._backpressure_event.set()  # Initially no backpressure
        
        self._stats = {
            "produced": 0,
            "consumed": 0,
            "backpressure_events": 0
        }
    
    @property
    def fill_level(self) -> float:
        """Current fill level (0.0 to 1.0)."""
        return self._queue.qsize() / self.maxsize
    
    @property
    def is_backpressured(self) -> bool:
        """Check if backpressure is active."""
        return not self._backpressure_event.is_set()
    
    async def put(self, item: T):
        """Put item with backpressure handling."""
        # Wait if backpressure is active
        await self._backpressure_event.wait()
        
        await self._queue.put(item)
        self._stats["produced"] += 1
        
        # Check if we should apply backpressure
        if self.fill_level >= self.high_water_mark:
            self._backpressure_event.clear()
            self._stats["backpressure_events"] += 1
            print(f"⚠️ Backpressure ON (fill: {self.fill_level:.1%})")
    
    async def get(self) -> T:
        """Get item and potentially release backpressure."""
        item = await self._queue.get()
        self._stats["consumed"] += 1
        
        # Check if we should release backpressure
        if self.is_backpressured and self.fill_level <= self.low_water_mark:
            self._backpressure_event.set()
            print(f"✓ Backpressure OFF (fill: {self.fill_level:.1%})")
        
        return item
    
    def task_done(self):
        """Mark task as done."""
        self._queue.task_done()
    
    async def join(self):
        """Wait for all tasks to complete."""
        await self._queue.join()
    
    def stats(self) -> dict:
        """Get queue statistics."""
        return {
            **self._stats,
            "current_size": self._queue.qsize(),
            "fill_level": self.fill_level,
            "backpressured": self.is_backpressured
        }


class ProducerConsumerPipeline:
    """
    Production-ready producer-consumer pipeline with:
    - Multiple producers and consumers
    - Backpressure handling
    - Graceful shutdown
    - Statistics
    """
    
    def __init__(
        self,
        queue_size: int = 100,
        num_producers: int = 2,
        num_consumers: int = 5
    ):
        self.queue = BackpressureQueue(maxsize=queue_size)
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        
        self._shutdown_event = asyncio.Event()
        self._producer_tasks: list = []
        self._consumer_tasks: list = []
        self._item_counter = 0
        self._lock = asyncio.Lock()
    
    async def _produce(self, producer_id: int, produce_func: Callable[[], Awaitable[T]]):
        """Producer coroutine."""
        while not self._shutdown_event.is_set():
            try:
                # Generate item
                data = await produce_func()
                
                async with self._lock:
                    self._item_counter += 1
                    item_id = self._item_counter
                
                item = WorkItem(id=item_id, data=data)
                
                # Put with backpressure handling
                await self.queue.put(item)
                
                print(f"[Producer-{producer_id}] Produced item {item_id}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Producer-{producer_id}] Error: {e}")
    
    async def _consume(self, consumer_id: int, process_func: Callable[[T], Awaitable[None]]):
        """Consumer coroutine."""
        while True:
            try:
                # Get item (will wait if empty)
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                print(f"[Consumer-{consumer_id}] Processing item {item.id}")
                
                try:
                    await process_func(item.data)
                finally:
                    self.queue.task_done()
                
                print(f"[Consumer-{consumer_id}] Completed item {item.id}")
            
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set():
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Consumer-{consumer_id}] Error: {e}")
    
    async def start(
        self,
        produce_func: Callable[[], Awaitable[T]],
        process_func: Callable[[T], Awaitable[None]]
    ):
        """Start the pipeline."""
        # Start consumers first
        self._consumer_tasks = [
            asyncio.create_task(self._consume(i, process_func))
            for i in range(self.num_consumers)
        ]
        
        # Start producers
        self._producer_tasks = [
            asyncio.create_task(self._produce(i, produce_func))
            for i in range(self.num_producers)
        ]
        
        print(f"Pipeline started: {self.num_producers} producers, {self.num_consumers} consumers")
    
    async def stop(self, timeout: float = 30.0):
        """Gracefully stop the pipeline."""
        print("Stopping pipeline...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel producers
        for task in self._producer_tasks:
            task.cancel()
        
        # Wait for queue to drain
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            print("Timeout waiting for queue to drain")
        
        # Cancel consumers
        for task in self._consumer_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(
            *self._producer_tasks,
            *self._consumer_tasks,
            return_exceptions=True
        )
        
        print(f"Pipeline stopped. Stats: {self.queue.stats()}")


async def pipeline_example():
    """Example usage of producer-consumer pipeline."""
    
    async def produce():
        """Generate work items."""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return {"value": random.randint(1, 100)}
    
    async def process(data):
        """Process work items (slow consumer)."""
        await asyncio.sleep(random.uniform(0.5, 1.0))
    
    pipeline = ProducerConsumerPipeline(
        queue_size=20,
        num_producers=3,
        num_consumers=2  # Deliberately slow to trigger backpressure
    )
    
    await pipeline.start(produce, process)
    
    # Run for 10 seconds
    await asyncio.sleep(10)
    
    await pipeline.stop()


# asyncio.run(pipeline_example())
```

## 7.4 Async Retry with Circuit Breaker

```python
import asyncio
from typing import TypeVar, Callable, Awaitable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import functools
import random

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: list = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(seconds=30),
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        if self._state != new_state:
            print(f"Circuit breaker: {self._state.value} -> {new_state.value}")
            self._stats.state_changes.append({
                "from": self._state.value,
                "to": new_state.value,
                "time": datetime.now()
            })
            self._state = new_state
            
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0
    
    async def _should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = datetime.now() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        await self._transition_to(CircuitState.HALF_OPEN)
                        return True
                
                self._stats.rejected_calls += 1
                return False
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        
        return False
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
            else:
                self._failure_count = 0
    
    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
    
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function through circuit breaker."""
        self._stats.total_calls += 1
        
        if not await self._should_allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is {self._state.value}"
            )
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for async retry with optional circuit breaker.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        circuit_breaker: Optional circuit breaker instance
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    if circuit_breaker:
                        return await circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                
                except CircuitBreakerOpenError:
                    raise  # Don't retry if circuit is open
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        # Add jitter
                        jitter = random.uniform(0, 0.1 * current_delay)
                        wait_time = current_delay + jitter
                        
                        print(f"Attempt {attempt + 1} failed: {e}. "
                              f"Retrying in {wait_time:.2f}s...")
                        
                        await asyncio.sleep(wait_time)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════

# Create a shared circuit breaker
api_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=timedelta(seconds=10)
)


@with_retry(
    max_attempts=3,
    delay=0.5,
    backoff=2.0,
    circuit_breaker=api_circuit_breaker
)
async def unreliable_api_call(endpoint: str) -> dict:
    """Simulated unreliable API call."""
    await asyncio.sleep(0.1)
    
    # Simulate 40% failure rate
    if random.random() < 0.4:
        raise ConnectionError(f"Failed to connect to {endpoint}")
    
    return {"status": "success", "endpoint": endpoint}


async def retry_example():
    """Demonstrate retry with circuit breaker."""
    
    for i in range(20):
        try:
            result = await unreliable_api_call(f"/api/endpoint/{i}")
            print(f"Call {i}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Call {i}: REJECTED - {e}")
        except Exception as e:
            print(f"Call {i}: FAILED - {e}")
        
        await asyncio.sleep(0.5)
    
    print(f"\nCircuit Breaker Stats: {api_circuit_breaker.stats()}")


# asyncio.run(retry_example())
```

## 7.5 Graceful Shutdown Handler

```python
import asyncio
import signal
from typing import Callable, Awaitable, List, Set
from dataclasses import dataclass
from enum import Enum
import functools


class ShutdownPhase(Enum):
    """Shutdown phases for ordered cleanup."""
    STOP_ACCEPTING = 1    # Stop accepting new work
    DRAIN_REQUESTS = 2    # Wait for in-flight requests
    CLOSE_CONNECTIONS = 3 # Close external connections
    CLEANUP = 4           # Final cleanup


@dataclass
class ShutdownTask:
    """Task to run during shutdown."""
    name: str
    phase: ShutdownPhase
    func: Callable[[], Awaitable[None]]
    timeout: float = 30.0


class GracefulShutdownManager:
    """
    Manages graceful shutdown with:
    - Signal handling (SIGTERM, SIGINT)
    - Phased shutdown
    - Timeout handling
    - In-flight request tracking
    """
    
    def __init__(self, shutdown_timeout: float = 60.0):
        self.shutdown_timeout = shutdown_timeout
        self._shutdown_event = asyncio.Event()
        self._shutdown_tasks: List[ShutdownTask] = []
        self._in_flight: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._is_shutting_down = False
    
    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down
    
    def register_shutdown_task(
        self,
        name: str,
        phase: ShutdownPhase,
        func: Callable[[], Awaitable[None]],
        timeout: float = 30.0
    ):
        """Register a task to run during shutdown."""
        self._shutdown_tasks.append(ShutdownTask(
            name=name,
            phase=phase,
            func=func,
            timeout=timeout
        ))
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
    
    async def _handle_signal(self, sig: signal.Signals):
        """Handle shutdown signal."""
        print(f"\nReceived signal {sig.name}, initiating graceful shutdown...")
        await self.shutdown()
    
    def track_request(self, task: asyncio.Task):
        """Track an in-flight request."""
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
    
    def request_tracker(self):
        """Decorator to track requests."""
        def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if self._is_shutting_down:
                    raise ServiceUnavailableError("Service is shutting down")
                
                task = asyncio.current_task()
                self.track_request(task)
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    async def wait_for_shutdown(self):
        """Wait until shutdown is requested."""
        await self._shutdown_event.wait()
    
    async def shutdown(self):
        """Execute graceful shutdown."""
        if self._is_shutting_down:
            return
        
        self._is_shutting_down = True
        self._shutdown_event.set()
        
        print("Starting graceful shutdown...")
        
        # Sort tasks by phase
        tasks_by_phase = {}
        for task in self._shutdown_tasks:
            if task.phase not in tasks_by_phase:
                tasks_by_phase[task.phase] = []
            tasks_by_phase[task.phase].append(task)
        
        # Execute phases in order
        for phase in ShutdownPhase:
            if phase in tasks_by_phase:
                print(f"\n[Shutdown Phase: {phase.name}]")
                
                for task in tasks_by_phase[phase]:
                    print(f"  Running: {task.name}")
                    try:
                        await asyncio.wait_for(
                            task.func(),
                            timeout=task.timeout
                        )
                        print(f"  ✓ {task.name} completed")
                    except asyncio.TimeoutError:
                        print(f"  ✗ {task.name} timed out")
                    except Exception as e:
                        print(f"  ✗ {task.name} failed: {e}")
            
            # Special handling for drain phase
            if phase == ShutdownPhase.DRAIN_REQUESTS and self._in_flight:
                print(f"\n  Waiting for {len(self._in_flight)} in-flight requests...")
                try:
                    await asyncio.wait_for(
                        self._wait_for_in_flight(),
                        timeout=self.shutdown_timeout
                    )
                    print("  ✓ All requests completed")
                except asyncio.TimeoutError:
                    print(f"  ✗ Timeout, {len(self._in_flight)} requests still in-flight")
                    # Cancel remaining requests
                    for task in self._in_flight:
                        task.cancel()
        
        print("\nGraceful shutdown complete")
    
    async def _wait_for_in_flight(self):
        """Wait for all in-flight requests to complete."""
        while self._in_flight:
            await asyncio.sleep(0.1)


class ServiceUnavailableError(Exception):
    """Raised when service is shutting down."""
    pass


# ═══════════════════════════════════════════════════════════════
# Complete application example
# ═══════════════════════════════════════════════════════════════

class Application:
    """Example application with graceful shutdown."""
    
    def __init__(self):
        self.shutdown_manager = GracefulShutdownManager(shutdown_timeout=30.0)
        self._server_running = False
    
    async def start(self):
        """Start the application."""
        # Register shutdown tasks
        self.shutdown_manager.register_shutdown_task(
            "Stop accepting connections",
            ShutdownPhase.STOP_ACCEPTING,
            self._stop_accepting
        )
        self.shutdown_manager.register_shutdown_task(
            "Close database pool",
            ShutdownPhase.CLOSE_CONNECTIONS,
            self._close_database
        )
        self.shutdown_manager.register_shutdown_task(
            "Flush caches",
            ShutdownPhase.CLEANUP,
            self._flush_caches
        )
        
        # Setup signal handlers
        self.shutdown_manager.setup_signal_handlers()
        
        self._server_running = True
        print("Application started")
        
        # Start background tasks
        asyncio.create_task(self._background_worker())
        
        # Run request handler
        asyncio.create_task(self._request_loop())
        
        # Wait for shutdown
        await self.shutdown_manager.wait_for_shutdown()
    
    @property
    def request_handler(self):
        """Decorated request handler."""
        @self.shutdown_manager.request_tracker()
        async def handle_request(request_id: int):
            print(f"Handling request {request_id}")
            await asyncio.sleep(2)  # Simulate work
            print(f"Request {request_id} complete")
        
        return handle_request
    
    async def _request_loop(self):
        """Simulate incoming requests."""
        request_id = 0
        while self._server_running:
            request_id += 1
            try:
                asyncio.create_task(self.request_handler(request_id))
            except ServiceUnavailableError:
                print(f"Request {request_id} rejected - shutting down")
            await asyncio.sleep(0.5)
    
    async def _background_worker(self):
        """Background worker."""
        while not self.shutdown_manager.is_shutting_down:
            await asyncio.sleep(1)
    
    async def _stop_accepting(self):
        """Stop accepting new connections."""
        self._server_running = False
        await asyncio.sleep(0.1)
    
    async def _close_database(self):
        """Close database connections."""
        await asyncio.sleep(0.5)  # Simulate closing connections
    
    async def _flush_caches(self):
        """Flush caches to disk."""
        await asyncio.sleep(0.3)  # Simulate flushing


async def application_example():
    """Run the example application."""
    app = Application()
    await app.start()


# To run: asyncio.run(application_example())
# Then press Ctrl+C to trigger graceful shutdown
```

---

# 8. Quick Reference Cheatsheets

## 8.1 Threading Cheatsheet

```python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         THREADING QUICK REFERENCE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import threading
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────────
# CREATING THREADS
# ─────────────────────────────────────────────────────────────────

# Method 1: Function target
t = threading.Thread(target=my_function, args=(arg1, arg2))
t.start()
t.join()

# Method 2: Subclass
class MyThread(threading.Thread):
    def run(self):
        # Thread code here
        pass

# Method 3: ThreadPoolExecutor (RECOMMENDED)
with ThreadPoolExecutor(max_workers=5) as executor:
    future = executor.submit(func, arg1, arg2)
    result = future.result()
    
    # Map multiple calls
    results = executor.map(func, iterable)


# ─────────────────────────────────────────────────────────────────
# THREAD PROPERTIES
# ─────────────────────────────────────────────────────────────────

t = threading.Thread(target=func, daemon=True)  # Daemon thread
t.name = "MyThread"                             # Set name
t.is_alive()                                    # Check if running
threading.current_thread()                      # Get current thread
threading.active_count()                        # Number of threads
threading.enumerate()                           # List all threads


# ─────────────────────────────────────────────────────────────────
# SYNCHRONIZATION PRIMITIVES
# ─────────────────────────────────────────────────────────────────

# Lock
lock = threading.Lock()
with lock:
    # Critical section
    pass

# RLock (reentrant)
rlock = threading.RLock()
with rlock:
    with rlock:  # Can acquire again
        pass

# Semaphore
sem = threading.Semaphore(3)  # Allow 3 concurrent
with sem:
    pass

# BoundedSemaphore (can't release more than acquired)
bsem = threading.BoundedSemaphore(3)

# Event
event = threading.Event()
event.set()                    # Set flag
event.clear()                  # Clear flag
event.is_set()                # Check flag
event.wait(timeout=5.0)       # Wait for flag

# Condition
cond = threading.Condition()
with cond:
    cond.wait()               # Wait for notification
    cond.wait_for(predicate)  # Wait for predicate
    cond.notify()             # Wake one waiter
    cond.notify_all()         # Wake all waiters

# Barrier
barrier = threading.Barrier(3)  # Wait for 3 threads
barrier.wait()                  # Wait at barrier


# ─────────────────────────────────────────────────────────────────
# THREAD-SAFE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

import queue

q = queue.Queue()             # FIFO
q = queue.LifoQueue()         # LIFO (stack)
q = queue.PriorityQueue()     # Priority

q.put(item)                   # Add (blocks if full)
q.put_nowait(item)           # Add (raises if full)
item = q.get()               # Remove (blocks if empty)
item = q.get_nowait()        # Remove (raises if empty)
q.task_done()                # Mark task complete
q.join()                     # Wait for all tasks


# ─────────────────────────────────────────────────────────────────
# THREAD-LOCAL STORAGE
# ─────────────────────────────────────────────────────────────────

local_data = threading.local()
local_data.value = "thread-specific"


# ─────────────────────────────────────────────────────────────────
# COMMON PATTERNS
# ─────────────────────────────────────────────────────────────────

# Pattern: Worker pool
def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        process(item)
        q.task_done()

# Pattern: Graceful shutdown
shutdown_event = threading.Event()

def worker():
    while not shutdown_event.is_set():
        if shutdown_event.wait(timeout=0.1):
            break
        do_work()

# Pattern: Lock ordering (prevent deadlock)
def transfer(acc1, acc2, amount):
    first, second = sorted([acc1, acc2], key=id)
    with first.lock, second.lock:
        # Safe transfer
        pass
```

## 8.2 Asyncio Cheatsheet

```python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ASYNCIO QUICK REFERENCE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio

# ─────────────────────────────────────────────────────────────────
# RUNNING ASYNC CODE
# ─────────────────────────────────────────────────────────────────

# Main entry point (Python 3.7+)
asyncio.run(main())

# Get running loop
loop = asyncio.get_running_loop()
loop = asyncio.get_event_loop()


# ─────────────────────────────────────────────────────────────────
# COROUTINES AND TASKS
# ─────────────────────────────────────────────────────────────────

# Define coroutine
async def my_coroutine():
    await asyncio.sleep(1)
    return "result"

# Create task (schedules coroutine)
task = asyncio.create_task(my_coroutine())
result = await task

# Task operations
task.cancel()                 # Request cancellation
task.cancelled()              # Check if cancelled
task.done()                   # Check if complete
task.result()                 # Get result (raises if not done)
task.exception()              # Get exception
task.add_done_callback(cb)    # Add completion callback


# ─────────────────────────────────────────────────────────────────
# CONCURRENT EXECUTION
# ─────────────────────────────────────────────────────────────────

# gather - run concurrently, collect all results
results = await asyncio.gather(
    coro1(),
    coro2(),
    coro3(),
    return_exceptions=True  # Don't raise, return exceptions
)

# wait - more control
done, pending = await asyncio.wait(
    tasks,
    timeout=10.0,
    return_when=asyncio.FIRST_COMPLETED  # or FIRST_EXCEPTION, ALL_COMPLETED
)

# as_completed - process as they finish
for coro in asyncio.as_completed(coroutines):
    result = await coro

# wait_for - single coroutine with timeout
result = await asyncio.wait_for(coro(), timeout=5.0)

# TaskGroup (Python 3.11+)
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(coro1())
    task2 = tg.create_task(coro2())


# ─────────────────────────────────────────────────────────────────
# TIMEOUTS (Python 3.11+)
# ─────────────────────────────────────────────────────────────────

async with asyncio.timeout(5.0):
    await long_operation()

async with asyncio.timeout_at(deadline):
    await long_operation()


# ─────────────────────────────────────────────────────────────────
# SYNCHRONIZATION PRIMITIVES
# ─────────────────────────────────────────────────────────────────

# Lock
lock = asyncio.Lock()
async with lock:
    # Critical section
    pass

# Semaphore
sem = asyncio.Semaphore(3)
async with sem:
    pass

# Event
event = asyncio.Event()
event.set()
event.clear()
await event.wait()

# Condition
cond = asyncio.Condition()
async with cond:
    await cond.wait()
    await cond.wait_for(predicate)
    cond.notify()
    cond.notify_all()

# Queue
queue = asyncio.Queue(maxsize=10)
await queue.put(item)
item = await queue.get()
queue.task_done()
await queue.join()


# ─────────────────────────────────────────────────────────────────
# CALLING BLOCKING CODE
# ─────────────────────────────────────────────────────────────────

# to_thread (Python 3.9+) - RECOMMENDED
result = await asyncio.to_thread(blocking_func, arg1, arg2)

# run_in_executor
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, blocking_func, arg)
result = await loop.run_in_executor(thread_pool, blocking_func, arg)
result = await loop.run_in_executor(process_pool, cpu_func, arg)


# ─────────────────────────────────────────────────────────────────
# CALLING ASYNC FROM SYNC (in another thread)
# ─────────────────────────────────────────────────────────────────

# Submit to running loop from another thread
future = asyncio.run_coroutine_threadsafe(coro(), loop)
result = future.result(timeout=10)


# ─────────────────────────────────────────────────────────────────
# ASYNC CONTEXT MANAGERS & ITERATORS
# ─────────────────────────────────────────────────────────────────

# Async context manager
async with resource:
    pass

class AsyncCM:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass

# Async iterator
async for item in async_iterable:
    pass

class AsyncIter:
    def __aiter__(self):
        return self
    async def __anext__(self):
        raise StopAsyncIteration

# Async generator
async def async_gen():
    yield value

# Async comprehension
results = [x async for x in async_gen()]


# ─────────────────────────────────────────────────────────────────
# CANCELLATION AND SHIELDING
# ─────────────────────────────────────────────────────────────────

# Handle cancellation
try:
    await some_operation()
except asyncio.CancelledError:
    # Cleanup
    raise  # Re-raise to propagate

# Shield from cancellation
await asyncio.shield(important_operation())


# ─────────────────────────────────────────────────────────────────
# SUBPROCESS
# ─────────────────────────────────────────────────────────────────

proc = await asyncio.create_subprocess_exec(
    'cmd', 'arg1',
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
)
stdout, stderr = await proc.communicate()


# ─────────────────────────────────────────────────────────────────
# STREAMS (Network I/O)
# ─────────────────────────────────────────────────────────────────

# TCP Client
reader, writer = await asyncio.open_connection('host', 8080)
data = await reader.read(100)
writer.write(b'data')
await writer.drain()
writer.close()
await writer.wait_closed()

# TCP Server
async def handle_client(reader, writer):
    pass

server = await asyncio.start_server(handle_client, 'host', 8080)
async with server:
    await server.serve_forever()
```

## 8.3 Decision Quick Reference

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CONCURRENCY DECISION QUICK REFERENCE                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ QUESTION                           │ ANSWER → USE                           │
├────────────────────────────────────┼────────────────────────────────────────┤
│ CPU-bound work?                    │ multiprocessing / ProcessPoolExecutor  │
│ I/O-bound, new project?            │ asyncio                                │
│ I/O-bound, legacy code?            │ threading                              │
│ Need 1000+ concurrent connections? │ asyncio                                │
│ Using blocking libraries?          │ threading (or asyncio + to_thread)     │
│ Simple background task?            │ threading (daemon thread)              │
│ WebSockets / real-time?            │ asyncio                                │
│ Database operations?               │ asyncio + async driver                 │
│ Web scraping (small scale)?        │ threading + requests                   │
│ Web scraping (large scale)?        │ asyncio + aiohttp                      │
│ GUI application?                   │ threading for background tasks         │
│ Microservices?                     │ asyncio                                │
└────────────────────────────────────┴────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMMON MISTAKES TO AVOID                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ❌ time.sleep() in async code          → Use asyncio.sleep()                │
│ ❌ requests library in async code      → Use aiohttp/httpx                  │
│ ❌ threading.Lock in async code        → Use asyncio.Lock()                 │
│ ❌ asyncio.Lock across threads         → Use threading.Lock()               │
│ ❌ CPU-bound in asyncio without thread → Use asyncio.to_thread()            │
│ ❌ Shared mutable state without lock   → Add proper synchronization         │
│ ❌ Not handling CancelledError         → Always handle or propagate         │
│ ❌ Forgetting await                    → Coroutine never executes           │
│ ❌ Creating tasks without reference    → Task may be garbage collected      │
│ ❌ Blocking the event loop             → Use run_in_executor/to_thread      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              IMPORT CHEATSHEET                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ # Threading                                                                 │
│ import threading                                                            │
│ from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor      │
│ import queue                                                                │
│                                                                             │
│ # Asyncio                                                                   │
│ import asyncio                                                              │
│                                                                             │
│ # Async Libraries                                                           │
│ import aiohttp          # HTTP client                                       │
│ import aiofiles         # File I/O                                          │
│ import asyncpg          # PostgreSQL                                        │
│ import aiomysql         # MySQL                                             │
│ import aiosqlite        # SQLite                                            │
│ import motor            # MongoDB                                           │
│ import aioredis         # Redis                                             │
│ import websockets       # WebSocket                                         │
│ import httpx            # HTTP (sync + async)                               │
│                                                                             │
│ # Async-Sync Bridge                                                         │
│ import janus            # Dual-interface queue                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Summary

This guide covered:

1. **Foundational Concepts**: GIL, concurrency vs parallelism, CPU vs I/O bound
2. **Threading**: All synchronization primitives, thread pools, best practices
3. **Asyncio**: Event loop, tasks, synchronization, async patterns
4. **I/O Blocking Reference**: Complete list of blocking operations and alternatives
5. **Comparison**: When to use threading vs asyncio
6. **Hybrid Approaches**: Combining async and sync code
7. **Real-World Patterns**: Production-ready implementations
8. **Quick Reference**: Cheatsheets for daily use

**Key Takeaways:**
- Use **asyncio** for new I/O-bound projects with high concurrency needs
- Use **threading** for legacy code or when using blocking libraries
- Use **multiprocessing** for CPU-bound work
- Always use **proper synchronization** for shared state
- **Never block the event loop** in asyncio code
- Use **`asyncio.to_thread()`** to run blocking code in async context
- Use **`asyncio.run_coroutine_threadsafe()`** to run async code from threads