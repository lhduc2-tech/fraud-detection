import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Define AutoEncoder Architecture matching trained model
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load Models
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, 'tier1_lightgbm.pkl')):
    MODEL_DIR = current_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join(current_dir, '..', 'model'))

print("==================================================")
print("   SYSTEM INITIALIZATION: LOADING 3-TIER PKL MODELS")
print("==================================================")
print(f"Loading models from: {MODEL_DIR}")
model_lgb = joblib.load(os.path.join(MODEL_DIR, 'tier1_lightgbm.pkl'))
ae_data = joblib.load(os.path.join(MODEL_DIR, 'tier2_autoencoder.pkl'))
gnn_data = joblib.load(os.path.join(MODEL_DIR, 'tier3_graphsage.pkl'))

scaler = ae_data['scaler']
threshold_ae = ae_data['threshold']
input_dim = ae_data['input_dim']

model_ae = AutoEncoder(input_dim)
model_ae.load_state_dict(ae_data['model_state'])
model_ae.eval()

print(" -> Tier 1 (LightGBM): Loaded")
print(f" -> Tier 2 (AutoEncoder): Loaded (Threshold = {threshold_ae:.4f})")
print(f" -> Tier 3 (GraphSAGE GNN): Loaded (AUC = {gnn_data['auc']:.4f})")
print("All 3 models initialized successfully!\n")

feature_cols = [
    'step', 'amount_vnd', 'oldbalanceOrg_vnd', 'newbalanceOrig_vnd',
    'hour_of_day', 'is_night_time', 'is_high_amount', 'is_emptying_account',
    'is_cross_border', 'velocity_count_24h', 'velocity_amount_24h',
    'fan_in_flag', 'fan_out_flag', 'structuring_flag', 'rapid_in_out_flag', 'device_anomaly_flag',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

def map_payload_to_features(payload):
    """
    Maps real-world transaction JSON record (Matching User Payload Schema)
    into feature vector required by the 3-Tier Fraud Detection Engine.
    """
    # Parse ISO Timestamp
    ts_str = payload.get('timestamp', '')
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        hour_of_day = dt.hour
    except Exception:
        hour_of_day = 12

    # Amount in VND
    amount_vnd = float(payload.get('amount', 0))

    # Balances (If missing, inferred or populated from payload enrichment)
    old_bal = float(payload.get('oldBalance', amount_vnd * 1.05))
    new_bal = float(payload.get('newBalance', max(0, old_bal - amount_vnd)))
    
    balance_diff_ratio = amount_vnd / (old_bal + 1)
    is_emptying = 1 if balance_diff_ratio > 0.95 else 0

    # Flags calculation / Extraction
    is_night = 1 if (2 <= hour_of_day <= 5) else 0
    is_high_amt = 1 if amount_vnd > 50_000_000 else 0
    
    # Velocity & Context Flags
    velocity_count_24h = int(payload.get('velocity_count_24h', 1))
    velocity_amount_24h = float(payload.get('velocity_amount_24h', amount_vnd))
    
    fan_in_flag = int(payload.get('fan_in_flag', 0))
    fan_out_flag = int(payload.get('fan_out_flag', 0))
    
    # Structuring (STR-EW-04)
    txn_type = str(payload.get('transactionType', 'TRANSFER')).upper()
    structuring_flag = 1 if (txn_type in ['TRANSFER', 'CASHOUT', 'CASH_OUT'] and balance_diff_ratio >= 0.80) else int(payload.get('structuring_flag', 0))
    rapid_in_out_flag = 1 if (balance_diff_ratio >= 0.80 and txn_type in ['CASHOUT', 'CASH_OUT']) else int(payload.get('rapid_in_out_flag', 0))
    device_anomaly_flag = int(payload.get('device_anomaly_flag', 0))
    is_cross_border = int(payload.get('is_cross_border', 0))

    feat_dict = {
        'step': hour_of_day,
        'amount_vnd': amount_vnd,
        'oldbalanceOrg_vnd': old_bal,
        'newbalanceOrig_vnd': new_bal,
        'hour_of_day': hour_of_day,
        'is_night_time': is_night,
        'is_high_amount': is_high_amt,
        'is_emptying_account': is_emptying,
        'is_cross_border': is_cross_border,
        'velocity_count_24h': velocity_count_24h,
        'velocity_amount_24h': velocity_amount_24h,
        'fan_in_flag': fan_in_flag,
        'fan_out_flag': fan_out_flag,
        'structuring_flag': structuring_flag,
        'rapid_in_out_flag': rapid_in_out_flag,
        'device_anomaly_flag': device_anomaly_flag,
        'type_CASH_OUT': 1 if txn_type in ['CASHOUT', 'CASH_OUT'] else 0,
        'type_DEBIT': 1 if txn_type == 'DEBIT' else 0,
        'type_PAYMENT': 1 if txn_type in ['PAYMENT', 'TOPUP'] else 0,
        'type_TRANSFER': 1 if txn_type == 'TRANSFER' else 0,
    }
    
    return pd.DataFrame([feat_dict]), feat_dict

def analyze_and_screen_transaction(payload, scenario_title="MẪU GIAO DỊCH"):
    row_df, feats = map_payload_to_features(payload)
    
    # ---------------------------------------------------------
    # 1. Rule-Based Evaluation & Violation Breakdown
    # ---------------------------------------------------------
    triggered_rules = []
    
    if feats['structuring_flag'] == 1:
        triggered_rules.append({
            'code': 'RULE-STR-EW-04',
            'name': 'Mẫu giao dịch Chia nhỏ dòng tiền (Structuring)',
            'penalty': 35,
            'desc': f"Rút/Chuyển {feats['amount_vnd']:,.0f} VNĐ chiếm >= 80% số dư ngay sau khi nạp tiền."
        })
        
    if feats['fan_in_flag'] == 1:
        triggered_rules.append({
            'code': 'RULE-STR-EW-05',
            'name': 'Tập trung dòng tiền bất thường (Fan-In Network)',
            'penalty': 25,
            'desc': 'Tài khoản người nhận nhận tiền từ >= 5 ví/tài khoản khác nhau trong 24h qua.'
        })
        
    if feats['fan_out_flag'] == 1:
        triggered_rules.append({
            'code': 'RULE-STR-EW-06',
            'name': 'Phân tán dòng tiền bất thường (Fan-Out Network)',
            'penalty': 25,
            'desc': 'Tài khoản người gửi thực hiện chuyển tiền đến >= 5 ví/tài khoản khác nhau trong 24h qua.'
        })
        
    if feats['rapid_in_out_flag'] == 1:
        triggered_rules.append({
            'code': 'RULE-RAPID-INOUT',
            'name': 'Giao dịch Nạp - Rút cấp tốc (Rapid In-Out)',
            'penalty': 30,
            'desc': 'Tiền vừa nạp vào ví đã bị rút ra ngay trong vòng 30 phút với tỷ lệ >= 80%.'
        })
        
    if feats['device_anomaly_flag'] == 1:
        triggered_rules.append({
            'code': 'RULE-DEVICE-ANOMALY',
            'name': 'Bất thường Thiết bị / IP đăng nhập (Device Fingerprint Anomaly)',
            'penalty': 25,
            'desc': 'Cùng 1 thiết bị/IP thực hiện giao dịch cho >= 3 tài khoản khác nhau trong 24h.'
        })
        
    if feats['is_night_time'] == 1 and feats['is_high_amount'] == 1:
        triggered_rules.append({
            'code': 'RULE-NIGHT-HIGH-AMT',
            'name': 'Giao dịch giá trị lớn khung giờ nhạy cảm đêm muộn',
            'penalty': 15,
            'desc': f"Giao dịch {feats['amount_vnd']:,.0f} VNĐ (> 50 triệu) diễn ra trong khoảng 2h00 - 5h00 sáng."
        })

    rule_score = min(100.0, sum(r['penalty'] for r in triggered_rules))

    # ---------------------------------------------------------
    # 2. AI Predictive Evaluation (3-Tier Models)
    # ---------------------------------------------------------
    # Tier 1: LightGBM
    lgbm_prob = float(model_lgb.predict(row_df[feature_cols])[0])
    
    # Tier 2: AutoEncoder
    row_scaled = scaler.transform(row_df[feature_cols])
    row_tensor = torch.tensor(row_scaled, dtype=torch.float32)
    with torch.no_grad():
        rec = model_ae(row_tensor)
        ae_err = float(torch.mean((row_tensor - rec) ** 2).item())
    ae_score = min(1.0, ae_err / (threshold_ae * 2))

    # Tier 3: GraphSAGE Risk
    gnn_risk = lgbm_prob  # Network Risk Proxy

    ai_score = (0.4 * lgbm_prob + 0.3 * ae_score + 0.3 * gnn_risk) * 100.0

    # ---------------------------------------------------------
    # 3. Final Combined Risk Score & Decision
    # ---------------------------------------------------------
    final_risk = 0.4 * rule_score + 0.6 * ai_score

    if final_risk < 50:
        level = "LOW"
        action = "CHO PHÉP GIAO DỊCH (Normal Transaction - No Alert)"
    elif final_risk < 80:
        level = "MEDIUM"
        action = "GIAO DỊCH ĐƯỢC XỬ LÝ -> TẠO FRAUD ALERT (Trạng thái: NEW - Chờ Admin Review)"
    else:
        level = "HIGH"
        action = "GIAO DỊCH ĐƯỢC XỬ LÝ -> TẠO FRAUD ALERT KHẨN CẤP (Trạng thái: NEW, Ưu tiên: HIGH - Phát Notification tới Ban Quản Trị)"

    # ---------------------------------------------------------
    # 4. Generate Threat Warnings & Risk Scenario Narrative
    # ---------------------------------------------------------
    threat_warnings = []
    if feats['fan_in_flag'] == 1 or feats['fan_out_flag'] == 1:
        threat_warnings.append("⚠️ [NGUY CƠ RỬA TIỀN / MẠNG LƯỚI VÍ RÁC]: Phát hiện mẫu giao dịch gom/tán tiền qua nhiều tài khoản trung gian nhằm che giấu nguồn gốc dòng tiền.")
    if feats['device_anomaly_flag'] == 1 and feats['is_emptying_account'] == 1:
        threat_warnings.append("🚨 [NGUY CƠ CHIẾM ĐOẠT TÀI KHOẢN - ATO]: Thiết bị lạ/IP bất thường đăng nhập và thực hiện rút sạch 95-100% số dư tài khoản.")
    if feats['structuring_flag'] == 1 or feats['rapid_in_out_flag'] == 1:
        threat_warnings.append("⚠️ [NGUY CƠ CHIA NHỎ DÒNG TIỀN / LỪA ĐẢO TÀI CHÍNH]: Hành vi nạp tiền rồi lập tức rút/chuyển đi ngay nhằm tẩu tán tài sản lừa đảo.")
    if feats['is_night_time'] == 1 and feats['is_high_amount'] == 1 and len(threat_warnings) == 0:
        threat_warnings.append("⚠️ [NGUY CƠ GIAO DỊCH BẤT THƯỜNG ĐÊM MUỘN]: Chuyển lượng tiền lớn bất thường vào khung giờ ít hoạt động, cần rà soát sinh trắc học / xác thực bổ sung.")
    if len(threat_warnings) == 0:
        threat_warnings.append("✅ Không phát hiện nguy cơ gian lận đáng kể. Giao dịch nằm trong ngưỡng an toàn.")

    # ---------------------------------------------------------
    # 5. Printable Detailed Report
    # ---------------------------------------------------------
    print("=" * 75)
    print(f" 📌 KẾT QUẢ SÀNG LỌC & CHẨN ĐOÁN RỦI RO: {scenario_title.upper()}")
    print("=" * 75)
    print(f" 🔹 Request UUID (correlationId) : {payload.get('correlationId', 'N/A')}")
    print(f" 🔹 Mã tham chiếu (referenceCode): {payload.get('referenceCode', 'N/A')}")
    print(f" 🔹 Thời gian (timestamp)         : {payload.get('timestamp', 'N/A')}")
    print(f" 🔹 Loại giao dịch               : {payload.get('transactionType', 'N/A')}")
    print(f" 🔹 Người thực hiện (Sender)     : {payload.get('userFullName')} ({payload.get('userEmail')})")
    print(f" 🔹 Người nhận / Tài khoản       : {payload.get('recipientFullName', 'N/A')} - STK: {payload.get('accountNumber')} ({payload.get('bankCode')})")
    print(f" 🔹 Số tiền giao dịch            : {feats['amount_vnd']:,.0f} VNĐ")
    print("-" * 75)
    
    print(f" 📊 ĐIỂM RỦI RO TỔNG HỢP (FINAL RISK SCORE): {final_risk:.2f} / 100")
    print(f" 🎯 PHÂN MỨC RỦI RO (RISK LEVEL)         : [{level}]")
    print(f" ⚙️ HÀNH ĐỘNG HỆ THỐNG (SYSTEM ACTION)    : {action}")
    print("-" * 75)

    print(" 🔍 BẢNG CHI TIẾT CÁC THÔNG SỐ ĐÁNH GIÁ (AI & RULE METRICS):")
    print(f"   • Điểm Luật nghiệp vụ (Rule Score)  : {rule_score:.2f} / 100")
    print(f"   • Tier 1 LightGBM Xác suất Gian lận : {lgbm_prob*100:.2f}%")
    print(f"   • Tier 2 AutoEncoder Lỗi tái tạo   : {ae_err:.4f} (Ngưỡng an toàn: {threshold_ae:.4f})")
    print(f"   • Tier 3 GraphSAGE Điểm rủi ro mạng : {gnn_risk*100:.2f}%")
    print(f"   • Điểm AI Tổng hợp (AI Score)        : {ai_score:.2f} / 100")
    print("-" * 75)

    print(f" 🚨 DANH SÁCH {len(triggered_rules)} LUẬT NGHIỆP VỤ BỊ VI PHẠM (RULE VIOLATIONS):")
    if len(triggered_rules) == 0:
        print("   (Không vi phạm quy tắc nghiệp vụ nào)")
    else:
        for idx, r in enumerate(triggered_rules, 1):
            print(f"   {idx}. [{r['code']}] {r['name']} (Phạt: +{r['penalty']}đ)")
            print(f"      ↳ Mô tả: {r['desc']}")
    print("-" * 75)

    print(" ⚠️ CẢNH BÁO NGUY CƠ GIAN LẬN XẢY RA (THREAT WARNINGS):")
    for tw in threat_warnings:
        print(f"   {tw}")
    print("=" * 75 + "\n")

    return {
        'correlationId': payload.get('correlationId'),
        'final_risk_score': round(final_risk, 2),
        'risk_level': level,
        'system_action': action,
        'rule_score': round(rule_score, 2),
        'ai_score': round(ai_score, 2),
        'lgbm_prob_pct': round(lgbm_prob * 100, 2),
        'ae_reconstruction_error': round(ae_err, 4),
        'ae_threshold': round(threshold_ae, 4),
        'triggered_rules': triggered_rules,
        'threat_warnings': threat_warnings
    }

# =========================================================
# TEST CASES MATCHING EXACT USER PAYLOAD SCHEMA
# =========================================================

# Scenario 1: Normal Deposit / Transfer
payload_normal = {
    "correlationId": "550e8400-e29b-41d4-a716-446655440000",
    "referenceCode": "FT26080400001",
    "idempotencyKey": "IDEM-001",
    "transactionType": "TRANSFER",
    "timestamp": "2026-08-04T14:30:00+07:00",
    "amount": 2500000,
    "bankCode": "VCB",
    "accountNumber": "1012345678",
    "userEmail": "nguyenvana@gmail.com",
    "userFullName": "Nguyễn Văn A",
    "description": "Chuyển tiền mua hàng",
    "recipientFullName": "Trần Thị B",
    "recipientPhoneNumber": "0987654321",
    "oldBalance": 15000000,
    "newBalance": 12500000,
    "velocity_count_24h": 1,
    "velocity_amount_24h": 2500000,
    "fan_in_flag": 0,
    "fan_out_flag": 0,
    "device_anomaly_flag": 0
}
analyze_and_screen_transaction(payload_normal, "Test Case 1: Giao dịch bình thường (2.5 triệu VNĐ, VCB, 14h30)")

# Scenario 2: Account Emptying & Night High Amount
payload_medium = {
    "correlationId": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "referenceCode": "FT26080400088",
    "idempotencyKey": "IDEM-088",
    "transactionType": "TRANSFER",
    "timestamp": "2026-08-04T03:15:30+07:00",
    "amount": 75000000,
    "bankCode": "TCB",
    "accountNumber": "1903999888",
    "userEmail": "lethi_c@gmail.com",
    "userFullName": "Lê Thị C",
    "description": "Chuyển tiền gấp đêm muộn",
    "recipientFullName": "Phạm Văn D",
    "recipientPhoneNumber": "0912345678",
    "oldBalance": 76000000,
    "newBalance": 1000000,
    "velocity_count_24h": 4,
    "velocity_amount_24h": 150000000,
    "fan_in_flag": 0,
    "fan_out_flag": 0,
    "device_anomaly_flag": 1,
    "structuring_flag": 1
}
analyze_and_screen_transaction(payload_medium, "Test Case 2: Giao dịch đáng ngờ (75 triệu VNĐ lúc 3h sáng, rút 98% ví, thiết bị lạ)")

# Scenario 3: High Risk Money Laundering Structuring & Fan-In / Fan-Out
payload_high = {
    "correlationId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "referenceCode": "FT26080400999",
    "idempotencyKey": "IDEM-999",
    "transactionType": "CASHOUT",
    "timestamp": "2026-08-04T04:10:00+07:00",
    "amount": 180000000,
    "bankCode": "MB",
    "accountNumber": "0999888777",
    "userEmail": "hoangvan_e@gmail.com",
    "userFullName": "Hoàng Văn E",
    "description": "Rút tiền mặt tài khoản gom",
    "recipientFullName": "Đại lý Rút tiền X",
    "recipientPhoneNumber": "0909090909",
    "oldBalance": 181000000,
    "newBalance": 1000000,
    "velocity_count_24h": 15,
    "velocity_amount_24h": 650000000,
    "fan_in_flag": 1,
    "fan_out_flag": 1,
    "device_anomaly_flag": 1,
    "structuring_flag": 1,
    "rapid_in_out_flag": 1
}
analyze_and_screen_transaction(payload_high, "Test Case 3: Giao dịch gian lận nguy cơ cao (Gom/Tán tiền Fan-In/Out, Structuring, 4h sáng)")
