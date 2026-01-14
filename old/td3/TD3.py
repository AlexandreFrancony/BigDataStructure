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

    def format_output(self):
        return (f"    Costs :\n"
                f"        time : {self.time_s:.6f} s\n"
                f"        carbon footprint : {self.carbon_g:.6f} g\n"
                f"        price : {self.price_eur:.6f} €")

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

def simulate_filter(db, coll_name, filter_key, selectivity, sharding_key=None, has_index=False):
    collection = db.collections[coll_name]
    output_count = collection.count * selectivity
    output_size = output_count * collection.doc_size
    
    algo = "Full scan"
    if has_index: algo = "Index"
    if sharding_key: algo = f"Shard / {algo}"

    if sharding_key and filter_key == sharding_key:
        nodes_involved = 1
        data_read = (collection.total_size_bytes / db.nb_servers) * (0.01 if has_index else 1.0)
        data_transfer = 0
    else:
        nodes_involved = db.nb_servers
        data_read = collection.total_size_bytes * (0.01 if has_index else 1.0)
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

def simulate_join(db, coll_outer, coll_inner, join_key, sharding_info=None):
    c_outer = db.collections[coll_outer]
    c_inner = db.collections[coll_inner]
    output_count = c_outer.count
    output_size = output_count * (c_outer.doc_size + c_inner.doc_size)
    
    algo = "Nested Loop"
    sh_keys = "None"

    if sharding_info:
        outer_sh = sharding_info.get("outer")
        inner_sh = sharding_info.get("inner")
        sh_keys = f"{outer_sh}, {inner_sh}"
        
        if outer_sh == join_key and inner_sh == join_key:
            algo = "Shard / Nested Loop"
            nodes_involved = db.nb_servers
            data_read = c_outer.total_size_bytes + c_inner.total_size_bytes
            data_transfer = 0
        else:
            algo = "Map/Reduce & Nested Loop"
            nodes_involved = db.nb_servers
            data_read = (c_outer.total_size_bytes + c_inner.total_size_bytes)
            data_transfer = min(c_outer.total_size_bytes, c_inner.total_size_bytes)
    else:
        nodes_involved = 1
        data_read = c_outer.total_size_bytes + c_inner.total_size_bytes
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
    # Setup DB1
    stats_DB1 = {"Product": 100000, "Stock": 21000, "Warehouse": 200, "OrderLine": 4_000_000_000, "Client": 10_000_000}
    root = "c:/Users/Léonard/Documents/travail/esilv/A5/Big data structure/"
    colls_DB1 = {n: Collection(n, f"{root}{n.lower()}.json", c) for n, c in stats_DB1.items()}
    db1 = Database("DB1", colls_DB1)

    # Setup DB5
    stats_DB5_prod = {"categories": 2, "orderlines": 40000}
    colls_DB5 = {
        "Product": Collection("Product", f"{root}DB5/product.json", 100000, stats_DB5_prod),
        "Stock": colls_DB1["Stock"], "Warehouse": colls_DB1["Warehouse"], "Client": colls_DB1["Client"]
    }
    db5 = Database("DB5", colls_DB5)

    print("=== DVL OPERATOR OPERATIONAL REPORT ===\n")

    # 1. Filter with sharding
    r1 = simulate_filter(db1, "Product", "IDP", 0.0001, "IDP", True)
    print_result(r1, "Filter Product on IDP (Sharded & Indexed)")

    # 2. Filter without sharding
    r2 = simulate_filter(db1, "Product", "brand", 0.02, None, False)
    print_result(r2, "Filter Product on brand (Full Scan)")

    # 3. Nested loop with sharding (Collocated)
    r3 = simulate_join(db1, "Product", "OrderLine", "IDP", {"outer": "IDP", "inner": "IDP"})
    print_result(r3, "Join Product and OrderLine on IDP (Collocated)")

    # 4. Nested loop without sharding (Non-sharded DB)
    r4 = simulate_join(db1, "Product", "OrderLine", "IDP", None)
    print_result(r4, "Join Product and OrderLine on IDP (Local)")

    # 5. DB5 Filter (Embedded)
    r5 = simulate_filter(db5, "Product", "IDP", 0.0001, "IDP", True)
    print_result(r5, "DB5: Filter Product (Embedded Orderlines)")

    # 6. Map/Reduce Join (Non-collocated sharding)
    r6 = simulate_join(db1, "OrderLine", "Client", "IDC", {"outer": "IDP", "inner": "IDC"})
    print_result(r6, "Join OrderLine(IDP) and Client(IDC) - Map/Reduce Shuffle")

if __name__ == "__main__":
    run_suite()
