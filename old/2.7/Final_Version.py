import json


# ============================================================
# 1. Load JSON Schema
# ============================================================

def load_schema(path):
    """Load a JSON Schema from file."""
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 2. Build Structure
# ============================================================

def build_structure(schema, stats=None):
    """Build an internal structure representation from a JSON Schema."""

    struct = {
        "title": schema.get("title", "Unknown"),
        "type": schema["type"],
        "fields": {}
    }

    properties = schema.get("properties", {})

    for name, prop in properties.items():

        field_type = prop.get("type")

        if field_type == "string":
            struct["fields"][name] = {"type": "string", "size": 20}

        elif field_type == "number":
            struct["fields"][name] = {"type": "number", "size": 8}

        elif field_type == "object":
            struct["fields"][name] = build_structure(prop, stats)

        elif field_type == "array":
            items = prop["items"]
            struct["fields"][name] = {
                "type": "array",
                "items": build_structure(items, stats)
            }

        else:
            struct["fields"][name] = {"type": field_type, "size": 16}

    return struct


# ============================================================
# 3. Compute sizes
# ============================================================

def compute_document_size(struct, stats=None):
    """Compute size of a document according to structure."""
    size = 0

    for field_name, field in struct["fields"].items():

        if field["type"] in ("string", "number"):
            size += field["size"]

        elif field["type"] == "object":
            size += compute_document_size(field, stats)

        elif field["type"] == "array":
            embedded_size = compute_document_size(field["items"], stats)

            # Special case: DB5 embedded OrderLines
            if "title" in field["items"] and field["items"]["title"] == "OrderLine":
                count = stats["Product"]["orderlines_per_product"]
            else:
                count = 1

            size += embedded_size * count

    return size


def compute_collection_size(struct, stats):
    """Compute total size of a collection."""
    name = struct["title"]

    if name not in stats or "count" not in stats[name]:
        raise ValueError(f"Missing count for '{name}' in stats")

    count = stats[name]["count"]
    size_doc = compute_document_size(struct, stats)
    total = size_doc * count

    return {
        "document_size": size_doc,
        "count": count,
        "total_bytes": total,
        "total_MB": total / (1024**2),
        "total_GB": total / (1024**3)
    }


def compute_database_size(db_structures, stats):
    """Compute total size of a database."""
    results = {}
    total = 0

    for name, struct in db_structures.items():
        r = compute_collection_size(struct, stats)
        results[name] = r
        total += r["total_bytes"]

    return {
        "collection_sizes": results,
        "total_bytes": total,
        "total_GB": total / (1024**3)
    }


# ============================================================
# 4. Sharding stats
# ============================================================

def compute_sharding_stats(collection, shard_key, stats, nb_servers=1000):
    """Compute sharding distribution."""
    total_docs = stats[collection]["count"]
    distinct_vals = stats[collection]["distinct"][shard_key]

    return {
        "collection": collection,
        "shard_key": shard_key,
        "docs_per_server": total_docs / nb_servers,
        "distinct_values_per_server": distinct_vals / nb_servers
    }


# ============================================================
# 5. Main runner
# ============================================================

def main():
    print("=== Big Data Package Runner ===")

    # Load schemas
    schema_product = load_schema("Product.json")
    schema_stock = load_schema("Stock.json")
    schema_warehouse = load_schema("Warehouse.json")
    schema_client = load_schema("Client.json")
    schema_orderline = load_schema("OrderLine.json")

    # Build structures
    sProduct = build_structure(schema_product, None)
    sStock = build_structure(schema_stock, None)
    sWarehouse = build_structure(schema_warehouse, None)
    sClient = build_structure(schema_client, None)
    sOrderLine = build_structure(schema_orderline, None)

    # DB1 stats
    stats_DB1 = {
        "Product": {"categories": 2, "suppliers": 1, "count": 100_000},
        "Stock": {"count": 105 * 200},
        "Warehouse": {"count": 200},
        "OrderLine": {"count": 4_000_000_000},
        "Client": {"count": 10_000_000}
    }

    # DB5 stats
    stats_DB5 = {
        "Product": {
            "categories": 2,
            "suppliers": 1,
            "orderlines_per_product": 40_000,
            "count": 100_000
        },
        "Stock": {"count": 105 * 200},
        "Warehouse": {"count": 200},
        "Client": {"count": 10_000_000}
    }

    # DB1 structure
    DB1 = {
        "Product": sProduct,
        "Stock": sStock,
        "Warehouse": sWarehouse,
        "OrderLine": sOrderLine,
        "Client": sClient
    }

    # DB5 structure
    DB5 = {
        "Product": sProduct,
        "Stock": sStock,
        "Warehouse": sWarehouse,
        "Client": sClient
    }

    # Compute DB1 / DB5 sizes
    res_DB1 = compute_database_size(DB1, stats_DB1)
    res_DB5 = compute_database_size(DB5, stats_DB5)

    print("\nDB1 total size (GB):", res_DB1["total_GB"])
    print("DB5 total size (GB):", res_DB5["total_GB"])

    # Sharding stats
    stats_sharding = {
        "Product": {
            "count": 100_000,
            "distinct": {"IDP": 100_000, "brand": 5_000}
        },
        "Stock": {
            "count": 105 * 200,
            "distinct": {"IDP": 105, "IDW": 200}
        },
        "OrderLine": {
            "count": 4_000_000_000,
            "distinct": {"IDC": 10_000_000, "IDP": 100_000}
        }
    }

    print("\n=== Sharding Statistics ===\n")

    cases = [
        ("Stock", "IDP"),
        ("Stock", "IDW"),
        ("OrderLine", "IDC"),
        ("OrderLine", "IDP"),
        ("Product", "IDP"),
        ("Product", "brand")
    ]

    for coll, key in cases:
        print(compute_sharding_stats(coll, key, stats_sharding))


# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":
    main()