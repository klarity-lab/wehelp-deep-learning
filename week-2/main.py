import math


# Task 1
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return math.dist((self.x, self.y), (other.x, other.y))


class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def slope(self):
        return (self.end.y - self.start.y) / (self.end.x - self.start.x)

    def is_parallel_to(self, other):
        # Parallel lines have the same slope.
        return self.slope() == other.slope()

    def is_perpendicular_to(self, other):
        # Perpendicular slopes multiply to -1.
        return self.slope() * other.slope() == -1


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return math.pi * self.radius**2

    def intersects(self, other):
        # Two circles meet when the distance between centers is between the
        # difference and the sum of the radii (touching counts as intersecting).
        distance = self.center.distance_to(other.center)
        return abs(self.radius - other.radius) <= distance <= self.radius + other.radius


class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def perimeter(self):
        total = 0
        points = self.vertices + [self.vertices[0]]
        for i in range(len(self.vertices)):
            total += points[i].distance_to(points[i + 1])
        return total


def run_task1():
    line_a = Line(Point(-6, 1), Point(2, 4))
    line_b = Line(Point(-6, -1), Point(2, 2))
    line_c = Line(Point(-1, 6), Point(-4, -4))

    circle_a = Circle(Point(6, 3), 2)
    circle_b = Circle(Point(8, 1), 1)

    polygon_a = Polygon([Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2)])

    print("Task 1")
    print(f"  Line A parallel to Line B?      {line_a.is_parallel_to(line_b)}")
    print(f"  Line C perpendicular to Line A? {line_c.is_perpendicular_to(line_a)}")
    print(f"  Area of Circle A:               {circle_a.area():.4f}")
    print(f"  Circle A intersects Circle B?   {circle_a.intersects(circle_b)}")
    print(f"  Perimeter of Polygon A:         {polygon_a.perimeter():.4f}")


# Task 2
class Enemy:
    def __init__(self, label, x, y, vector, life=10):
        self.label = label
        self.x = x
        self.y = y
        self.vector = vector
        self.life = life

    def is_alive(self):
        return self.life > 0

    def move(self):
        self.x += self.vector[0]
        self.y += self.vector[1]


class Tower:
    def __init__(self, label, x, y, attack, attack_range):
        self.label = label
        self.x = x
        self.y = y
        self.attack = attack
        self.attack_range = attack_range

    def enemies_in_range(self, enemies):
        in_range = []
        for enemy in enemies:
            distance = math.dist((self.x, self.y), (enemy.x, enemy.y))
            if enemy.is_alive() and distance <= self.attack_range:
                in_range.append(enemy)
        return in_range


def run_simulation(enemies, towers, turns=10):
    for _ in range(turns):
        for enemy in enemies:
            if enemy.is_alive():
                enemy.move()
        for tower in towers:
            # Re-check range per tower so enemies killed earlier this turn are skipped.
            for enemy in tower.enemies_in_range(enemies):
                enemy.life -= tower.attack


def run_task2():
    enemies = [
        Enemy("E1", -10, 2, (2, -1)),
        Enemy("E2", -8, 0, (3, 1)),
        Enemy("E3", -9, -1, (3, 0)),
    ]
    towers = [
        Tower("T1", -3, 2, 1, 2),
        Tower("T2", -1, -2, 1, 2),
        Tower("T3", 4, 2, 1, 2),
        Tower("T4", 7, 0, 1, 2),
        Tower("A1", 1, 1, 2, 4),
        Tower("A2", 4, -3, 2, 4),
    ]

    run_simulation(enemies, towers, 10)

    print("\nTask 2 (after 10 turns)")
    for enemy in enemies:
        status = "alive" if enemy.is_alive() else "dead"
        print(
            f"  label: {enemy.label}, "
            f"final position: ({enemy.x}, {enemy.y}), "
            f"life: {enemy.life} ({status})"
        )


run_task1()
run_task2()
