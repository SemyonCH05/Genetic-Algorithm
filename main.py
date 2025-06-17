import random
import copy
import openrouteservice
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

points_data = []
priorities_map = {}
duration_matrix = []
time_limit_seconds = 0


MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

def calculate_fitness(individual_route_indices):
    global priorities_map, duration_matrix, time_limit_seconds

    if not individual_route_indices:
        return 0, 0

    total_priority = 0
    total_duration = 0

    for point_idx in individual_route_indices:
        total_priority += priorities_map.get(point_idx, 0)

    if len(individual_route_indices) > 1:
        for i in range(len(individual_route_indices) - 1):
            p1_idx = individual_route_indices[i]
            p2_idx = individual_route_indices[i + 1]
            total_duration += duration_matrix[p1_idx][p2_idx]

    if total_duration > time_limit_seconds:
        return 0, total_duration

    return total_priority, total_duration


def create_individual():
    # создание маршрута
    global points_data
    num_total_points = len(points_data)
    if num_total_points == 0:
        return []

    num_points_in_route = random.randint(1, num_total_points)

    available_indices = list(range(num_total_points))
    route_indices = random.sample(available_indices, num_points_in_route)

    return route_indices


def initialize_population(size):
    return [create_individual() for _ in range(size)]


def selection(population_with_fitness):
    selected_parents = []
    for _ in range(len(population_with_fitness)):
        tournament = random.sample(population_with_fitness, TOURNAMENT_SIZE)
        winner = max(tournament, key=lambda x: x[1][0])
        selected_parents.append(winner[0])
    return selected_parents


def crossover(parent1_route, parent2_route):
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1_route), copy.deepcopy(parent2_route)

    p1 = copy.deepcopy(parent1_route)
    p2 = copy.deepcopy(parent2_route)

    child1 = []
    child2 = []

    if len(p1) >= 2:
        start_idx, end_idx = sorted(random.sample(range(len(p1)), 2))
        segment1 = p1[start_idx: end_idx + 1]
        child1.extend(segment1)
        for point_idx in p2:
            if point_idx not in child1:
                child1.append(point_idx)
    elif len(p1) == 1:
        child1.extend(p1)
        for point_idx in p2:
            if point_idx not in child1:
                child1.append(point_idx)
    else:
        child1 = copy.deepcopy(p2)

    if len(p2) >= 2:
        start_idx, end_idx = sorted(random.sample(range(len(p2)), 2))
        segment2 = p2[start_idx: end_idx + 1]
        child2.extend(segment2)

        for point_idx in p1:
            if point_idx not in child2:
                child2.append(point_idx)
    elif len(p2) == 1:
        child2.extend(p2)
        for point_idx in p1:
            if point_idx not in child2:
                child2.append(point_idx)
    else:
        child2 = copy.deepcopy(p1)

    return child1, child2


def mutate(individual_route):
    if random.random() > MUTATION_RATE:
        return individual_route

    mutated_route = copy.deepcopy(individual_route)
    if not mutated_route:
        return mutated_route

    mutation_type = random.choice(['swap', 'insert', 'add', 'remove'])
    num_total_points = len(points_data)

    if mutation_type == 'swap' and len(mutated_route) >= 2:
        idx1, idx2 = random.sample(range(len(mutated_route)), 2)
        mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]

    elif mutation_type == 'insert' and len(mutated_route) >= 2:
        point_to_move_idx = random.randrange(len(mutated_route))
        point_val = mutated_route.pop(point_to_move_idx)
        new_pos_idx = random.randrange(len(mutated_route) + 1)
        mutated_route.insert(new_pos_idx, point_val)

    elif mutation_type == 'add' and len(mutated_route) < num_total_points:
        available_points_to_add = [i for i in range(num_total_points) if i not in mutated_route]
        if available_points_to_add:
            point_to_add = random.choice(available_points_to_add)
            insert_pos = random.randrange(len(mutated_route) + 1)
            mutated_route.insert(insert_pos, point_to_add)

    elif mutation_type == 'remove' and len(mutated_route) > 0:
        remove_idx = random.randrange(len(mutated_route))
        mutated_route.pop(remove_idx)

    return mutated_route


def genetic_algorithm():
    global points_data, priorities_map, duration_matrix, time_limit_seconds

    if not points_data:
        print("Нет данных о точках для запуска ГА.")
        return [], 0, 0

    population = initialize_population(50)
    best_overall_individual = []
    best_overall_fitness = -1
    best_overall_duration = float('inf')

    for generation in range(100):
        population_with_fitness = []
        for individual in population:
            fitness, path_duration = calculate_fitness(individual)
            population_with_fitness.append((individual, (fitness, path_duration)))

        population_with_fitness.sort(key=lambda x: x[1][0], reverse=True)

        current_best_individual, (current_best_fitness, current_best_duration) = population_with_fitness[0]

        if current_best_fitness > best_overall_fitness:
            best_overall_individual = copy.deepcopy(current_best_individual)
            best_overall_fitness = current_best_fitness
            best_overall_duration = current_best_duration
        new_population = []

        # Элитизм: добавление лучших из предыдущего поколения
        for i in range(ELITISM_COUNT):
            new_population.append(population_with_fitness[i][0])

        selected_parents = selection(population_with_fitness)

        num_offspring_needed = 50 - ELITISM_COUNT
        offspring_list = []

        while len(offspring_list) < num_offspring_needed:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            child1, child2 = crossover(parent1, parent2)
            offspring_list.append(mutate(child1))
            if len(offspring_list) < num_offspring_needed:
                offspring_list.append(mutate(child2))

        new_population.extend(offspring_list[:num_offspring_needed])
        population = new_population

    return best_overall_individual, best_overall_fitness, best_overall_duration


def main():
    global points_data, priorities_map, duration_matrix, time_limit_seconds

    map_types = ["driving-car", "driving-hgv", "foot-walking",
    "foot-hiking", "cycling-regular", "cycling-road",
    "cycling-safe", "cycling-mountain", "cycling-tour", "cycling-electric"
    ]
    map_type = map_types[0]
    time_limit_seconds = 20 * 60

    points_input = [
        {'coords': (55.786255, 49.163872), 'priority': 8},
        {'coords': (55.792095, 49.152398), 'priority': 4},
        {'coords': (55.792044, 49.121827), 'priority': 12},
        {'coords': (55.790845, 49.117552), 'priority': 29},
        {'coords': (55.798131, 49.106281), 'priority': 15},
        {'coords': (55.788991, 49.135711), 'priority': 7},
        {'coords': (55.832394, 49.051177), 'priority': 50}
    ]
    points_data_corrected = [
        {'coords': (p['coords'][1], p['coords'][0]), 'priority': p['priority']}  # (lon, lat)
        for p in points_input
    ]
    points_data = points_data_corrected

    coordinates_for_api = [p['coords'] for p in points_data]
    priorities_map = {i: p["priority"] for i, p in enumerate(points_data)}

    client = openrouteservice.Client(key='5b3ce3597851110001cf624849b43a0299834c198e8502bd4df10db6')


    matrix_response = client.distance_matrix(
        locations=coordinates_for_api,
        profile=map_type,
        metrics=['duration']
    )
    duration_matrix = matrix_response['durations']

    best_solution_indices, best_fitness, best_duration = genetic_algorithm()

    print(f"Лучший маршрут (индексы точек): {best_solution_indices}")
    print(f"Максимальный суммарный приоритет: {best_fitness}")
    print(f"Время прохождения маршрута: {best_duration / 60:.2f} минут ({best_duration:.0f} сек)")

    if best_solution_indices:
        route_coordinates = [points_data[i]['coords'] for i in best_solution_indices]
        if len(route_coordinates) < 2:
            if len(route_coordinates) ==1:
                print(f"Посещается одна точка: {points_data[best_solution_indices[0]]['coords']}")
            return

        print(f"Координаты для построения маршрута: {route_coordinates}")


        route_geojson = client.directions(
            coordinates=route_coordinates,
            profile=map_type,
            format='geojson'
        )

        gdf = gpd.GeoDataFrame.from_features(route_geojson['features'], crs='EPSG:4326').to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, linewidth=4, color='blue', zorder=2)

        # Отображение всех точек из points_data
        all_points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([p['coords'][0] for p in points_data],
                                        [p['coords'][1] for p in points_data]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        all_points_gdf.plot(ax=ax, color='black', markersize=50, alpha=0.5, zorder=1, label="Все точки")

        # Отображение точек маршрута
        route_points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([c[0] for c in route_coordinates],
                                        [c[1] for c in route_coordinates]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        route_points_gdf.plot(ax=ax, color='red', markersize=100, zorder=3, label="Точки маршрута")

        for i, idx in enumerate(best_solution_indices):
            point_geom = route_points_gdf.geometry.iloc[i]
            ax.annotate(f"{i + 1}",
                        (point_geom.x, point_geom.y),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha='left',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_axis_off()
        ax.legend()
        plt.title(f"Оптимальный маршрут: Приоритет {best_fitness}, Время {best_duration / 60:.1f} мин")
        plt.show()
    else:
        print("Не удалось найти подходящий маршрут.")


if __name__ == '__main__':
    main()