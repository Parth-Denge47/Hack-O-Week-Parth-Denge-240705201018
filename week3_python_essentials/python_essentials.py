"""Week 3: Python essentials - OOP, dunder methods, comprehensions, generators.

Author: Parth Denge | PRN: 240705201018
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402


# ---------- OOP: a small inventory with inheritance ----------
class Item:
    def __init__(self, name: str, price: float, qty: int = 0):
        if price < 0 or qty < 0:
            raise ValueError("price and qty must be non-negative")
        self.name, self.price, self.qty = name, price, qty

    def total_value(self) -> float:
        return self.price * self.qty

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, price={self.price}, qty={self.qty})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Item) and (self.name, self.price) == (other.name, other.price)

    def __hash__(self) -> int:
        return hash((self.name, self.price))


class PerishableItem(Item):
    def __init__(self, name, price, qty=0, days_left=7):
        super().__init__(name, price, qty)
        self.days_left = days_left

    def total_value(self) -> float:
        # items about to expire are discounted by 50%
        factor = 0.5 if self.days_left <= 2 else 1.0
        return super().total_value() * factor


class Inventory:
    def __init__(self):
        self._items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        if item.name in self._items:
            self._items[item.name].qty += item.qty
        else:
            self._items[item.name] = item

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def __getitem__(self, name: str) -> Item:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def total_value(self) -> float:
        return sum(i.total_value() for i in self)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)


# ---------- comprehensions ----------
def squares_of_evens(n: int) -> list[int]:
    return [x * x for x in range(n) if x % 2 == 0]


def word_lengths(words: list[str]) -> dict[str, int]:
    return {w: len(w) for w in words}


def unique_initials(words: list[str]) -> set[str]:
    return {w[0].upper() for w in words if w}


def flatten(matrix: list[list[int]]) -> list[int]:
    return [v for row in matrix for v in row]


# ---------- generators ----------
def fibonacci(limit: int):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


def read_in_chunks(data: list, size: int):
    """Stream a sequence in fixed-size chunks without copying it all at once."""
    for i in range(0, len(data), size):
        yield data[i:i + size]


def running_average(values):
    total = 0.0
    for n, v in enumerate(values, 1):
        total += v
        yield total / n


def main():
    banner("Week 3 - Python Essentials")
    inv = Inventory()
    inv.add(Item("Notebook", 40.0, 25))
    inv.add(Item("Pen", 10.0, 100))
    inv.add(PerishableItem("Milk", 60.0, 10, days_left=1))
    inv.add(Item("Pen", 10.0, 50))
    print("Items in inventory:", len(inv))
    for it in inv:
        print("  ", it, "-> value", it.total_value())
    print("Total inventory value:", inv.total_value())
    print("Point add:", Point(1, 2) + Point(3, 4))
    print("Squares of evens < 10:", squares_of_evens(10))
    print("Word lengths:", word_lengths(["data", "science", "python"]))
    print("Unique initials:", sorted(unique_initials(["apple", "avocado", "banana"])))
    print("Flatten:", flatten([[1, 2], [3], [4, 5]]))
    print("Fibonacci < 100:", list(fibonacci(100)))
    print("Chunks of 3:", list(read_in_chunks(list(range(8)), 3)))
    print("Running average:", [round(v, 2) for v in running_average([2, 4, 6, 8])])


if __name__ == "__main__":
    main()
