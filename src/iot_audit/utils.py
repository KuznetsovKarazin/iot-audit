
from __future__ import annotations
import pandas as pd
from typing import List, Dict

NETWORK_COLS = [
    "src_ip","src_port","dst_ip","dst_port","proto","service","duration",
    "src_bytes","dst_bytes","conn_state","missed_bytes","src_pkts","src_ip_bytes",
    "dst_pkts","dst_ip_bytes"
]

DNS_COLS = [
    "dns_query","dns_qclass","dns_qtype","dns_rcode","dns_AA","dns_RD","dns_RA","dns_rejected"
]

SSL_COLS = [
    "ssl_version","ssl_cipher","ssl_resumed","ssl_established","ssl_subject","ssl_issuer"
]

HTTP_COLS = [
    "http_trans_depth","http_method","http_uri","http_version","http_request_body_len",
    "http_response_body_len","http_status_code","http_user_agent","http_orig_mime_types",
    "http_resp_mime_types"
]

WEIRD_COLS = ["weird_name","weird_addl","weird_notice"]
TARGET_COLS = ["label","type"]

ALL_GROUPS = {
    "network": NETWORK_COLS,
    "dns": DNS_COLS,
    "ssl": SSL_COLS,
    "http": HTTP_COLS,
    "weird": WEIRD_COLS,
    "target": TARGET_COLS
}

def infer_present_groups(df: pd.DataFrame) -> Dict[str, list]:
    return {k:[c for c in v if c in df.columns] for k,v in ALL_GROUPS.items()}

def basic_summary(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum().to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        "shape": list(df.shape),
        "nulls": nulls,
        "dtypes": dtypes,
    }
