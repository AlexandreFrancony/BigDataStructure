# Big Data Structure

This repository contains the coursework and project code for the **Big Data Structure** module at ESILV. Its main goal is to model and simulate how different NoSQL data models and infrastructures impact **performance**, **carbon footprint**, and **financial cost**.

## Objectives

- Design several denormalized JSON-based data models (DB1–DB5) starting from a relational schema.
- Implement a generic structure to:
  - Represent collections from JSON Schemas.
  - Compute document, collection, and database sizes.
  - Estimate sharding distributions over a cluster of servers.
- Simulate query execution costs on a distributed NoSQL cluster, including:
  - Filter queries (with and without sharding and indexes).
  - Join queries (local vs. sharded, with possible data shuffle).
  - Aggregate queries (map/reduce-style, with or without sharding).

## What the code does

- Parses JSON Schemas and statistics to build in-memory representations of collections and databases.
- Computes document sizes using approximate type sizes (string, number, date, long string, arrays, etc.) and key overhead.
- Estimates:
  - Total data volume per collection and per database.
  - Data distribution with different sharding keys.
- Provides operator-like functions to **simulate**:
  - Filters (`simulate_filter`)
  - Joins (`simulate_join`)
  - Aggregations (`simulate_aggregate`)
- For each operator, computes:
  - Execution time (I/O + network)
  - Energy consumption and resulting carbon footprint
  - Monetary cost based on a given kWh price

The repository is organized so that previous TDs are archived, while reusable code (schemas, JSON data, and operators) can be plugged together to evaluate different denormalization and sharding strategies on a realistic workload.