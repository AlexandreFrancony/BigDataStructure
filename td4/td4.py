from pathlib import Path
import json
import math

# ============================================================
# Constants and Constants for Costs
# ============================================================

TYPE_SIZES = {
    "string": 80,
    "number": 8,
    "integer": 8,
    "date": 20,
    "longstring": 200
}
KEY_OVERHEAD = 12

NB_SERVERS_DEFAULT = 1000
IO_SPEED_BYTES_S = 100 * 1024 * 1024
NETWORK_SPEED_BYTES_S = 10 * 1024 * 1024
CARBON_INTENSITY_G_KWH = 50
POWER_CONSUMPTION_W = 200
KWH_PRICE = 0.15

# ============================================================
# Data Structures
# ============================================

class Collection:
    def __init__(self, name, schema_path, count, stats=None):
        self.name = name
        with open(schema_path, "r") as f:
            self.schema = json.load(f)
        self.count = count
        self.stats = stats or {}
        self.doc_size = self._compute_doc_size(self.schema)

    def _compute_doc_size(self, schema_node):
        size = 0
        properties = schema_node.get("properties", {})
        for key, info in properties.items():
            field_type = info.get("type", "object")
            if field_type in TYPE_SIZES:
                size += KEY_OVERHEAD + TYPE_SIZES[field_type]
            elif field_type == "object":
                size += KEY_OVERHEAD + self._compute_doc_size(info)
            elif field_type == "array":
                items = info.get("items", {})
                item_type = items.get("type", "object")
                cardinality = self.stats.get(key, 1)
                item_size = self._compute_doc_size(items) if item_type == "object" else (KEY_OVERHEAD + TYPE_SIZES.get(item_type, 80))
                size += KEY_OVERHEAD + (cardinality * item_size)
        return size

    @property
    def total_size_bytes(self):
        return self.count * self.doc_size

class Database:
    def __init__(self, name, collections, nb_servers=NB_SERVERS_DEFAULT):
        self.name = name
        self.collections = collections
        self.nb_servers = nb_servers

# ============================================================
# Cost Calculation Engine
# ============================================================

class CostResult:
    def __init__(self, time_s, carbon_g, price_eur):
        self.time_s = time_s
        self.carbon_g = carbon_g
        self.price_eur = price_eur

    def add(self, other):
        return CostResult(
            self.time_s + other.time_s,
            self.carbon_g + other.carbon_g,
            self.price_eur + other.price_eur,
        )

    def format_output(self):
        # coût par heure équivalent, en supposant 1 exécution
        cost_per_hour = (3600 / self.time_s) * self.price_eur if self.time_s > 0 else 0
        # coût pour 1000 exécutions
        cost_per_1000 = self.price_eur * 1000

        return (
            "    Costs :\n"
            f"        time : {self.time_s:.6f} s\n"
            f"        carbon footprint : {self.carbon_g:.6f} g\n"
            f"        price : {self.price_eur:.6f} € "
            f"(for 1 execution, {cost_per_1000:.6f} € for 1000 exec.)\n"
            f"        equivalent hourly rate : {cost_per_hour:.6f} €/h\n"
        )


def calculate_costs(data_read_bytes, data_transfer_bytes, nodes_involved):
    time_read = data_read_bytes / IO_SPEED_BYTES_S
    time_transfer = data_transfer_bytes / NETWORK_SPEED_BYTES_S
    total_time = (time_read + time_transfer) / nodes_involved
    energy_kwh = (POWER_CONSUMPTION_W * (total_time / 3600.0)) * nodes_involved / 1000.0
    carbon = energy_kwh * CARBON_INTENSITY_G_KWH
    price = energy_kwh * KWH_PRICE
    return CostResult(total_time, carbon, price)

# ============================================================
# Operators
# ============================================================

def simulate_filter(db, coll_name, filter_key, selectivity, sharding_key=None, has_index=False, input_data=None):
    # Use input_data if piped from a previous operator
    if input_data:
        count = input_data["OutputDocs"]
        doc_size = input_data["OutputSize"] / count if count > 0 else 0
        total_size = input_data["OutputSize"]
    else:
        collection = db.collections[coll_name]
        count = collection.count
        doc_size = collection.doc_size
        total_size = collection.total_size_bytes
        
    output_count = count * selectivity
    output_size = output_count * doc_size
    
    algo = "Full scan"
    if has_index: algo = "Index"
    if sharding_key: algo = f"Shard / {algo}"

    if sharding_key and filter_key == sharding_key:
        nodes_involved = 1
        data_read = (total_size / db.nb_servers) * (0.01 if has_index else 1.0)
        data_transfer = 0
    else:
        nodes_involved = db.nb_servers
        data_read = total_size * (0.01 if has_index else 1.0)
        data_transfer = output_size if sharding_key else 0

    costs = calculate_costs(data_read, data_transfer, nodes_involved)
    
    return {
        "Database": db.name,
        "Sharding keys": sharding_key or "None",
        "Index": filter_key if has_index else "None",
        "Algo": algo,
        "OutputDocs": int(output_count),
        "OutputSize": output_size,
        "Costs": costs
    }

def simulate_aggregate(db, coll_name, group_keys, sharding_key=None, input_data=None, distinct_values=None):
    if input_data:
        count = input_data["OutputDocs"]
        total_size = input_data["OutputSize"]
    else:
        collection = db.collections[coll_name]
        count = collection.count
        total_size = collection.total_size_bytes

    # If distinct_values (D) is not provided, we estimate it or use stats
    # Instructions: number of output documents / distinct key values
    output_count = distinct_values if distinct_values else (count * 0.1) # Placeholder 10%
    output_size = output_count * 100 # Aggregated docs are usually smaller (grouped keys + sums)

    algo = "Local Aggregate"
    if sharding_key:
        if group_keys == sharding_key:
            algo = "Shard / Local Aggregate"
            nodes_involved = db.nb_servers
            data_read = total_size
            data_transfer = 0 # No shuffle needed
        else:
            algo = "Map/Reduce Aggregate (Shuffle)"
            nodes_involved = db.nb_servers
            data_read = total_size
            data_transfer = total_size * 0.5 # Shuffle overhead (estimated)
    else:
        nodes_involved = 1
        data_read = total_size
        data_transfer = 0

    costs = calculate_costs(data_read, data_transfer, nodes_involved)

    return {
        "Database": db.name,
        "Sharding keys": sharding_key or "None",
        "Index": "None",
        "Algo": algo,
        "OutputDocs": int(output_count),
        "OutputSize": output_size,
        "Costs": costs
    }

def simulate_join(db, coll_outer, coll_inner, join_key, sharding_info=None, outer_input=None, inner_input=None):
    if outer_input:
        c_outer_count = outer_input["OutputDocs"]
        c_outer_size = outer_input["OutputSize"]
        c_outer_doc_size = c_outer_size / c_outer_count if c_outer_count > 0 else 0
    else:
        c_outer = db.collections[coll_outer]
        c_outer_count = c_outer.count
        c_outer_size = c_outer.total_size_bytes
        c_outer_doc_size = c_outer.doc_size

    if inner_input:
        c_inner_count = inner_input["OutputDocs"]
        c_inner_size = inner_input["OutputSize"]
        c_inner_doc_size = c_inner_size / c_inner_count if c_inner_count > 0 else 0
    else:
        c_inner = db.collections[coll_inner]
        c_inner_count = c_inner.count
        c_inner_size = c_inner.total_size_bytes
        c_inner_doc_size = c_inner.doc_size
        
    output_count = c_outer_count
    output_size = output_count * (c_outer_doc_size + c_inner_doc_size)
    
    algo = "Nested Loop"
    sh_keys = "None"

    if sharding_info:
        outer_sh = sharding_info.get("outer")
        inner_sh = sharding_info.get("inner")
        sh_keys = f"{outer_sh}, {inner_sh}"
        
        if outer_sh == join_key and inner_sh == join_key:
            algo = "Shard / Nested Loop"
            nodes_involved = db.nb_servers
            data_read = c_outer_size + c_inner_size
            data_transfer = 0
        else:
            algo = "Map/Reduce & Nested Loop"
            nodes_involved = db.nb_servers
            data_read = c_outer_size + c_inner_size
            data_transfer = min(c_outer_size, c_inner_size)
    else:
        nodes_involved = 1
        data_read = c_outer_size + c_inner_size
        data_transfer = 0

    costs = calculate_costs(data_read, data_transfer, nodes_involved)

    return {
        "Database": db.name,
        "Sharding keys": sh_keys,
        "Index": "None",
        "Algo": algo,
        "OutputDocs": int(output_count),
        "OutputSize": output_size,
        "Costs": costs
    }

def print_result(res, name):
    print(f"Query : {name}")
    print(f"Database : {res['Database']}")
    print(f"Sharding keys : {res['Sharding keys']}")
    print(f"Index : {res['Index']}")
    print(f"Algo : {res['Algo']}")
    # print(f"Documents : {res['OutputDocs']} ({res['OutputSize']/(1024**2):.2f} MB)")
    print(res['Costs'].format_output())
    print("-" * 40)

def run_suite():
    # Dossier du fichier td4.py
    td4_dir = Path(__file__).resolve().parent
    # Racine du projet = parent de td4/
    project_root = td4_dir.parent
    data_root = project_root / "data"

    # Setup DB1
    stats_DB1 = {
        "Product": 100000,
        "Stock": 21000,
        "Warehouse": 200,
        "OrderLine": 4_000_000_000,
        "Client": 10_000_000,
    }

    # Fichiers JSON de DB1 dans data/
    colls_DB1 = {
        n: Collection(n, data_root / f"{n.lower()}.json", c)
        for n, c in stats_DB1.items()
    }
    db1 = Database("DB1", colls_DB1)

    # Setup DB5
    stats_DB5_prod = {"categories": 2, "orderlines": 40000}
    db5_root = data_root / "DB5"

    colls_DB5 = {
        "Product": Collection("Product", db5_root / "product.json", 100000, stats_DB5_prod),
        "Stock": colls_DB1["Stock"],
        "Warehouse": colls_DB1["Warehouse"],
        "Client": colls_DB1["Client"],
    }
    db5 = Database("DB5", colls_DB5)

    print("=== DVL OPERATOR OPERATIONAL REPORT ===\n")

    # 1. Filter with sharding
    r1 = simulate_filter(db1, "Stock", "IDW", 0.0001, sharding_key="IDW", has_index=True)
    print_result(r1, "Query : Q1 - Filter Stock on IDW (Sharded & Indexed)")

    # 2. Filter without sharding
    r2 = simulate_filter(db1, "Product", "brand", 0.02, None, False)
    print_result(r2, "Query : Q2 - Filter Product on brand (Full Scan)")

    # 3. Nested loop with sharding (Collocated)
    r3 = simulate_join(db1, "Product", "OrderLine", "IDP", {"outer": "IDP", "inner": "IDP"})
    print_result(r3, "Query : Q3 - Join Product and OrderLine on IDP (Collocated)")

    # 4. Nested loop without sharding (Non-sharded DB)
    r4 = simulate_join(db1, "Product", "OrderLine", "IDP", None)
    print_result(r4, "Query : Q4 - Join Product and OrderLine on IDP (Local)")

    # 5. DB5 Filter (Embedded)
    r5 = simulate_filter(db5, "Product", "IDP", 0.0001, "IDP", True)
    print_result(r5, "Query : Q5 - DB5: Filter Product (Embedded Orderlines)")

    # 7. Q6: 100 most ordered product names (Aggregate + Join)
    print("Query : Q6 - 100 most ordered products (Aggregate + Join)")
    # Step A: Aggregate OrderLine by IDP (distinct IDP = 100,000)
    q6a = simulate_aggregate(db1, "OrderLine", "IDP", "IDC", distinct_values=100000)
    # Step B: Join Product with result
    q6b = simulate_join(db1, "Product", None, "IDP", {"outer": "IDP", "inner": "IDP"}, inner_input=q6a)
    q6_total_cost = q6a["Costs"].add(q6b["Costs"])
    print(f"Database : DB1\nSharding keys : IDP, IDC\nAlgo : Map/Reduce Aggregate + Shard Join")
    print(q6_total_cost.format_output())
    print("-" * 40)

    # 8. Q7: Most ordered product by customer 125 (Filter + Aggregate + Join)
    print("Query : Q7 - Most ordered product by customer 125 (Filter + Aggregate + Join)")
    # Step A: Filter OrderLine by idClient=125 (selectivity 1/10m)
    q7a = simulate_filter(db1, "OrderLine", "idClient", 1/10_000_000, "IDC", False)
    # Step B: Aggregate result by IDP
    q7b = simulate_aggregate(db1, "OrderLine", "IDP", "IDC", input_data=q7a, distinct_values=400) # Assumption: 400 products per client
    # Step C: Join Product with result
    q7c = simulate_join(db1, "Product", None, "IDP", {"outer": "IDP", "inner": "IDP"}, inner_input=q7b)
    q7_total_cost = q7a["Costs"].add(q7b["Costs"]).add(q7c["Costs"])
    print(f"Database : DB1\nSharding keys : IDP, IDC\nAlgo : Shard Filter + M/R Aggregate + Shard Join")
    print(q7_total_cost.format_output())
    print("-" * 40)

    # 9. Challenge: details of products stored in warehouse 1
    print("Query : Challenge - Products stored in warehouse 1")

    # Step A: filter Warehouse by IDW = 1
    # 200 warehouses -> selectivity ≈ 1/200
    q_ch_a = simulate_filter(
        db1,
        "Warehouse",
        "IDW",
        selectivity=1/200,
        sharding_key="IDW",
        has_index=True,  # index on IDW
    )

    # Step B: join Warehouse result with Stock on IDW
    # Sharding strategy: Stock(#idP), Warehouse(#idW)
    # -> non collocated join on IDW (Map/Reduce & Nested Loop)
    q_ch_b = simulate_join(
        db1,
        coll_outer="Warehouse",
        coll_inner="Stock",
        join_key="IDW",
        sharding_info={"outer": "IDW", "inner": "IDP"},
        outer_input=q_ch_a,
        inner_input=None,  # Stock taken as full collection
    )

    # Total costs
    q_ch_total = q_ch_a["Costs"].add(q_ch_b["Costs"])

    print("Database : DB1")
    print("Sharding keys : IDW, IDP")
    print("Algo : Shard Filter (Warehouse) + Map/Reduce & Nested Loop Join (Warehouse–Stock on IDW)")
    print(q_ch_total.format_output())
    print("-" * 40)

if __name__ == "__main__":
    run_suite()
