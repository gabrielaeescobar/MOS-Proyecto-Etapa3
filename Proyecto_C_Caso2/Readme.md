# Proyecto C Caso 2 - National logistics

## Overview
This project focuses on solving the routing of transportation vehicles that deliver the clients request all across the country problem using data provided in the form of three CSV files: `clients.csv`, `vehicles.csv`, `depots.csv`, `stations.csv`. These datasets contain essential information about the delivery network, including client demands, available vehicles, and depot locations.

## Key considerations
- **ADDITIONAL VARIABLES**: As for the variables, such as the fuel price, fuel efficiency, truck maintenance cost, electricity cost, energy consumption, drone maintenance cost, drivers wage, drone operator wage are not included in the datasets. You are encouraged to research and include them in your analysis.

## Data Description

### 1. `clients.csv`
This file contains information about the clients that require deliveries. The key columns include:
- **LocationID**: Unique identifier for each Location.
- **ClientID**: Unique identifier for each client.
- **City/Municipality**: The city or municipality where the client is located.
- **Demand**: The quantity of goods required by the client in kg.

### 2. `vehicles.csv`
This file provides details about the vehicles available for deliveries. The key columns include:
- **VehicleID**: Unique identifier for each vehicle.
- **Capacity**: Maximum load the vehicle can carry in kg.
- **Range**: The maximum distance the vehicle can travel on a single charge or tank of fuel in km.
### 3. `depots.csv`
This file contains information about the depots where vehicles are stationed. The key columns include:
- **LocationID**: Unique identifier for each Location.
- **DepotID**: Unique identifier for each depot.
- **Latitude**: Latitude coordinate of the client's location.
- **Longitude**: Longitude coordinate of the client's location.

## 4. `stations.csv`
This file contains information about the where the fuel stations are located. The key columns include:
- **LocationID**: Unique identifier for each Location.
- **StationID**: Unique identifier for each station.
- **Latitude**: Latitude coordinate of the client's location.
- **Longitude**: Longitude coordinate of the client's location.
- **FuelPrice**: The price of fuel at the station COP/gal.