# LED Matrix Controller

[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/worgarside/led-matrix-controller)

This service is intended to be run on a Raspberry Pi and receives messages from a MQTT broker to control an LED matrix. I currently control my matrix from [Home Assistant](https://github.com/worgarside/home-assistant).

## Content

All content has the following parameters:

### Clock

Literally just a clock.

| ![Clock](./readme/clock.gif) |
|-----------------------------|
| *The clock, with a demonstration of the position/scale parameters being modified* |

#### Parameters

| Name | Description |
|------|-------------|
| `scale` | Scale of the clock, can only be 1 or 2 |
| `x_pos` | X position of the content |
| `y_pos` | Y position of the content |

### Raining Grid

A cellular automata that simulates the current rainfall. My Home Assistant instance sends messages to the Pi via MQTT to update the rain chance parameter.

It can also display a plant growth simulation, where the plants grow according to a Home Assistant-controlled configuration and decay back to nothing when the rain drops below a given threshold.

| ![Raining Grid](./readme/raining-grid.gif) |
|--------------------------------------------|
| *The raining grid simulation (in real time)* |

| ![Raining Grid with Plants](./readme/raining-grid-plants.gif) |
|--------------------------------------------|
| *The raining grid simulation with plants (in 8x speed)* |

#### Parameters

| Name | Description |
|------|-------------|
| `rain_chance` | The chance of rain, between 0 and 100 |
| `rain_speed` | The time period between rain cells, in ticks |
| `splash_speed` | The time period between rain cells, in ticks |
| `plant_limit` | The maximum number of plants on the grid |
| `plant_growth_chance` | The chance of a plant growing when rain hits the bottom/the end of a plant, between 0 and 100 |
| `distance_between_plants` | The distance between plants, in cells |
| `plant_decay_propagation_speed` | The speed at which a plant decays, in ticks |
| `leaf_growth_chance` | The chance of a leaf growing in a valid location on any given tick, between 0 and 100 |


### Sorting Algorithms

A visualisation of a number of sorting algorithms.

| ![Sorting Algorithms](./readme/sorting-algorithms.gif) |
|-------------------------------------------------------|
| *A random selection of the sorting algorithms* |

#### Parameters

| Name | Description |
|------|-------------|
| `algorithm` | The algorithm to use |
| `completion_display_time` | The time to display the sorted array for, in seconds |
| `iterations` | The number of iterations to run the sorting algorithm for before expiring the content |
| `randomize_algorithm` | Whether to randomize the algorithm on each iteration |
