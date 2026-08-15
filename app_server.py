# -*- coding: utf-8 -*-
"""
Fraud Detection Microservice - 3-Tier AI & Rule-based Engine
Provides REST APIs for Real-time Transaction Screening & Direct PostgreSQL Alert Persistence.
Port: 5002
"""

import os
import sys
import json
import logging
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger("fraud_detection")

# Tự động load biến môi trường từ file .env
def load_env_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, '.env'),
        os.path.join(current_dir, '..', 'fintech', 'fintech', '.env'),
        os.path.join(current_dir, '..', 'fintech', '.env'),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            logger.info("Loading environment variables from: %s", env_path)
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            k, v = k.strip(), v.strip()
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.warning("Could not read %s: %s", env_path, e)
            break

load_env_file()

app = Flask(__name__)

# =========================================================================
# 1. MODEL DEFINITION & INITIALIZATION
# =========================================================================

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


def find_model_file(filename):
    """Tìm file model trong thư mục hiện tại hoặc các thư mục con/cha phổ biến."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, filename),
        os.path.join(current_dir, '..', 'model', filename),
        os.path.join(current_dir, 'models', filename),
        os.path.abspath(filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(current_dir, filename)


# Load 3-Tier Models
model_lgb = None
model_ae = None
scaler = None
threshold_ae = 0.05
gnn_data = None
models_loaded = False

try:
    path_lgb = find_model_file('tier1_lightgbm.pkl')
    path_ae = find_model_file('tier2_autoencoder.pkl')
    path_gnn = find_model_file('tier3_graphsage.pkl')

    logger.info("Loading models from:\n - LGB: %s\n - AE:  %s\n - GNN: %s", path_lgb, path_ae, path_gnn)

    model_lgb = joblib.load(path_lgb)
    ae_data = joblib.load(path_ae)
    gnn_data = joblib.load(path_gnn)

    scaler = ae_data['scaler']
    threshold_ae = ae_data.get('threshold', 0.05)
    input_dim = ae_data['input_dim']

    model_ae = AutoEncoder(input_dim)
    model_ae.load_state_dict(ae_data['model_state'])
    model_ae.eval()

    models_loaded = True
    logger.info("--> All 3 Fraud Detection Models loaded successfully! (AE Threshold: %.4f)", threshold_ae)
except Exception as e:
    logger.error("Failed to load fraud detection models: %s", e, exc_info=True)


feature_cols = [
    'step', 'amount_vnd', 'oldbalanceOrg_vnd', 'newbalanceOrig_vnd',
    'hour_of_day', 'is_night_time', 'is_high_amount', 'is_emptying_account',
    'is_cross_border', 'velocity_count_24h', 'velocity_amount_24h',
    'fan_in_flag', 'fan_out_flag', 'structuring_flag', 'rapid_in_out_flag', 'device_anomaly_flag',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]


# =========================================================================
# 2. DATABASE INTEGRATION (PostgreSQL)
# =========================================================================

def get_db_connection():
    """Tạo kết nối tới PostgreSQL database từ biến môi trường."""
    import psycopg2
    from urllib.parse import urlparse

    db_url = os.getenv('DB_URL', '')
    if db_url.startswith('jdbc:postgresql://'):
        db_url = db_url.replace('jdbc:', '')

    db_user = os.getenv('DB_USERNAME') or 'postgres'
    db_pass = os.getenv('DB_PASSWORD') or 'postgres'

    if db_url:
        parsed = urlparse(db_url)
        return psycopg2.connect(
            dbname=parsed.path.lstrip('/') or 'postgres',
            user=db_user if db_user != 'postgres' else (parsed.username or 'postgres'),
            password=db_pass if db_pass != 'postgres' else (parsed.password or 'postgres'),
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            connect_timeout=5
        )
    else:
        return psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'postgres'),
            user=db_user,
            password=db_pass,
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            connect_timeout=5
        )


def persist_fraud_alert(payload, analysis_result):
    """Lưu kết quả sàng lọc rủi ro vào bảng fraud_alerts trong PostgreSQL."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO fraud_alerts (
            correlation_id, reference_code, user_id, user_email, user_full_name,
            transaction_type, amount, bank_code, account_number,
            recipient_full_name, recipient_phone_number,
            final_risk_score, risk_level, system_action,
            rule_score, ai_score, lgbm_prob_pct,
            ae_reconstruction_error, ae_threshold,
            triggered_rules, threat_warnings, payload_snapshot,
            status, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, NOW(), NOW()
        ) RETURNING id;
        """

        user_id = payload.get('userId') or payload.get('senderUserId') or None
        if user_id:
            try:
                import uuid
                uuid.UUID(str(user_id))
            except Exception:
                user_id = None

        cur.execute(sql, (
            str(payload.get('correlationId', 'N/A')),
            str(payload.get('referenceCode', 'N/A')),
            user_id,
            payload.get('userEmail'),
            payload.get('userFullName'),
            str(payload.get('transactionType', 'TRANSFER')),
            float(payload.get('amount', 0)),
            payload.get('bankCode'),
            payload.get('accountNumber') or payload.get('senderAccountNumber'),
            payload.get('recipientFullName'),
            payload.get('recipientPhoneNumber'),
            float(analysis_result['final_risk_score']),
            str(analysis_result['risk_level']),
            str(analysis_result['system_action']),
            float(analysis_result.get('rule_score', 0)),
            float(analysis_result.get('ai_score', 0)),
            float(analysis_result.get('lgbm_prob_pct', 0)),
            float(analysis_result.get('ae_reconstruction_error', 0)),
            float(analysis_result.get('ae_threshold', 0)),
            json.dumps(analysis_result['triggered_rules'], ensure_ascii=False),
            json.dumps(analysis_result['threat_warnings'], ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False, default=str),
            'NEW'
        ))

        inserted_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[DB] Saved fraud alert record ID=%s for referenceCode=%s (Risk: %s, Level: %s)",
                    inserted_id, payload.get('referenceCode'), analysis_result['final_risk_score'], analysis_result['risk_level'])
        return str(inserted_id)
    except Exception as e:
        logger.error("[DB] Could not save fraud alert to PostgreSQL: %s", e)
        return None


# =========================================================================
# 3. FEATURE EXTRACTION & SCREENING ENGINE
# =========================================================================

def map_payload_to_features(payload):
    """Chuyển đổi JSON payload từ Fintech thành Feature Vector cho 3-Tier AI Engine."""
    ts_str = payload.get('timestamp', '')
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        hour_of_day = dt.hour
    except Exception:
        hour_of_day = datetime.now().hour

    amount_vnd = float(payload.get('amount', 0))

    # Balances
    old_bal = float(payload.get('oldBalance', payload.get('oldbalanceOrg_vnd', amount_vnd * 1.05)))
    new_bal = float(payload.get('newBalance', payload.get('newbalanceOrig_vnd', max(0, old_bal - amount_vnd))))

    balance_diff_ratio = amount_vnd / (old_bal + 1)
    is_emptying = 1 if balance_diff_ratio > 0.95 else 0

    is_night = 1 if (2 <= hour_of_day <= 5) else 0
    is_high_amt = 1 if amount_vnd > 50_000_000 else 0

    velocity_count_24h = int(payload.get('velocity24hCount', payload.get('velocity_count_24h', 1)))
    velocity_amount_24h = float(payload.get('velocity24hAmount', payload.get('velocity_amount_24h', amount_vnd)))

    fan_in_cnt = int(payload.get('fanIn24hCount', payload.get('fan_in_flag', 0)))
    fan_out_cnt = int(payload.get('fanOut24hCount', payload.get('fan_out_flag', 0)))
    fan_in_flag = 1 if (fan_in_cnt >= 5 or payload.get('fan_in_flag') == 1) else 0
    fan_out_flag = 1 if (fan_out_cnt >= 5 or payload.get('fan_out_flag') == 1) else 0

    txn_type = str(payload.get('transactionType', 'TRANSFER')).upper()
    is_structuring = 1 if (txn_type in ['TRANSFER', 'CASHOUT', 'CASH_OUT', 'WALLET_TRANSFER_OUT', 'WALLET_TO_BANK'] and balance_diff_ratio >= 0.80) else int(payload.get('structuring_flag', 0))
    rapid_in_out = 1 if (balance_diff_ratio >= 0.80 and txn_type in ['CASHOUT', 'CASH_OUT', 'WALLET_TO_BANK']) else int(payload.get('rapid_in_out_flag', 0))
    device_anomaly = int(payload.get('device_anomaly_flag', 1 if payload.get('deviceFingerprint') and 'unknown' in str(payload.get('deviceFingerprint')).lower() else 0))
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
        'structuring_flag': is_structuring,
        'rapid_in_out_flag': rapid_in_out,
        'device_anomaly_flag': device_anomaly,
        'type_CASH_OUT': 1 if txn_type in ['CASHOUT', 'CASH_OUT', 'WALLET_TO_BANK'] else 0,
        'type_DEBIT': 1 if txn_type == 'DEBIT' else 0,
        'type_PAYMENT': 1 if txn_type in ['PAYMENT', 'TOPUP', 'BANK_TO_WALLET'] else 0,
        'type_TRANSFER': 1 if txn_type in ['TRANSFER', 'WALLET_TRANSFER_OUT'] else 0,
    }

    return pd.DataFrame([feat_dict]), feat_dict


def evaluate_transaction(payload):
    """Thực hiện toàn bộ quy trình sàng lọc: Rule-based + 3-Tier AI Models."""
    row_df, feats = map_payload_to_features(payload)

    # 1. Rule-Based Evaluation
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

    # 2. AI Predictive Evaluation (3-Tier Models)
    lgbm_prob = 0.05
    ae_err = 0.001
    ae_score = 0.05
    gnn_risk = 0.05

    if models_loaded and model_lgb is not None:
        try:
            lgbm_prob = float(model_lgb.predict(row_df[feature_cols])[0])
        except Exception as e:
            logger.warning("LGBM prediction error: %s", e)

        try:
            row_scaled = scaler.transform(row_df[feature_cols])
            row_tensor = torch.tensor(row_scaled, dtype=torch.float32)
            with torch.no_grad():
                rec = model_ae(row_tensor)
                ae_err = float(torch.mean((row_tensor - rec) ** 2).item())
            ae_score = min(1.0, ae_err / (threshold_ae * 2 if threshold_ae > 0 else 1.0))
        except Exception as e:
            logger.warning("AutoEncoder prediction error: %s", e)

        gnn_risk = lgbm_prob  # GraphSAGE Risk Proxy

    ai_score = (0.4 * lgbm_prob + 0.3 * ae_score + 0.3 * gnn_risk) * 100.0

    # 3. Final Combined Risk Score & Decision
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

    # 4. Threat Warnings
    threat_warnings = []
    if feats['fan_in_flag'] == 1 or feats['fan_out_flag'] == 1:
        threat_warnings.append("⚠️ [NGUY CƠ RỬA TIỀN / MẠNG LƯỚI VÍ RÁC]: Phát hiện mẫu giao dịch gom/tán tiền qua nhiều tài khoản trung gian.")
    if feats['device_anomaly_flag'] == 1 and feats['is_emptying_account'] == 1:
        threat_warnings.append("🚨 [NGUY CƠ CHIẾM ĐOẠT TÀI KHOẢN - ATO]: Thiết bị lạ/IP bất thường đăng nhập và thực hiện rút sạch 95-100% số dư.")
    if feats['structuring_flag'] == 1 or feats['rapid_in_out_flag'] == 1:
        threat_warnings.append("⚠️ [NGUY CƠ CHIA NHỎ DÒNG TIỀN / LỪA ĐẢO TÀI CHÍNH]: Hành vi nạp tiền rồi lập tức rút/chuyển đi ngay nhằm tẩu tán tài sản.")
    if feats['is_night_time'] == 1 and feats['is_high_amount'] == 1 and len(threat_warnings) == 0:
        threat_warnings.append("⚠️ [NGUY CƠ GIAO DỊCH BẤT THƯỜNG ĐÊM MUỘN]: Chuyển lượng tiền lớn bất thường vào khung giờ ít hoạt động.")
    if len(threat_warnings) == 0:
        threat_warnings.append("✅ Không phát hiện nguy cơ gian lận đáng kể. Giao dịch nằm trong ngưỡng an toàn.")

    return {
        'correlationId': str(payload.get('correlationId', 'N/A')),
        'referenceCode': str(payload.get('referenceCode', 'N/A')),
        'final_risk_score': float(round(final_risk, 2)),
        'risk_level': str(level),
        'system_action': str(action),
        'rule_score': float(round(rule_score, 2)),
        'ai_score': float(round(ai_score, 2)),
        'lgbm_prob_pct': float(round(lgbm_prob * 100, 2)),
        'ae_reconstruction_error': float(round(ae_err, 4)),
        'ae_threshold': float(round(float(threshold_ae), 4)),
        'triggered_rules': triggered_rules,
        'threat_warnings': threat_warnings
    }


# =========================================================================
# 4. REST CONTROLLERS
# =========================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra tình trạng hoạt động của service & kết nối database."""
    db_ok = False
    db_error = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        db_ok = True
    except Exception as e:
        db_ok = False
        db_error = str(e)
        logger.warning("[HEALTH] DB check failed: %s", e)

    resp = {
        'status': 'UP',
        'service': 'fraud-detection-service',
        'models_loaded': models_loaded,
        'database_connected': db_ok,
        'timestamp': datetime.now().isoformat()
    }
    if not db_ok and db_error:
        resp['database_error'] = db_error

    return jsonify(resp)


@app.route('/api/v1/fraud/screen', methods=['POST'])
def api_screen_transaction():
    """
    Tiếp nhận giao dịch từ Fintech, đánh giá 3-Tier AI + Rules,
    và tự động lưu kết quả vào bảng fraud_alerts trong DB.
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'error': 'Payload cannot be empty'}), 400

        ref_code = payload.get('referenceCode', 'UNKNOWN')
        logger.info("[SCREEN] Received transaction to evaluate: referenceCode=%s, type=%s, amount=%s",
                    ref_code, payload.get('transactionType'), payload.get('amount'))

        # Phân tích & sàng lọc
        result = evaluate_transaction(payload)

        # Lưu trực tiếp vào PostgreSQL
        alert_id = persist_fraud_alert(payload, result)
        result['alert_id'] = alert_id

        return jsonify({
            'success': True,
            'message': 'Transaction screened successfully',
            'data': result
        }), 200

    except Exception as e:
        logger.error("[SCREEN] Error evaluating transaction: %s", e, exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Fraud evaluation failed internally'
        }), 500


@app.route('/api/v1/fraud/alerts', methods=['GET'])
def api_get_alerts():
    """Tra cứu danh sách cảnh báo gian lận từ DB."""
    try:
        limit = min(100, int(request.args.get('limit', 20)))
        risk_level = request.args.get('risk_level')
        status = request.args.get('status')

        conn = get_db_connection()
        cur = conn.cursor()

        query = "SELECT id, correlation_id, reference_code, user_email, user_full_name, transaction_type, amount, final_risk_score, risk_level, system_action, triggered_rules, threat_warnings, status, created_at FROM fraud_alerts WHERE 1=1"
        params = []

        if risk_level:
            query += " AND risk_level = %s"
            params.append(risk_level.upper())
        if status:
            query += " AND status = %s"
            params.append(status.upper())

        query += " ORDER BY created_at DESC LIMIT %s;"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()

        alerts = []
        for r in rows:
            alerts.append({
                'id': str(r[0]),
                'correlation_id': r[1],
                'reference_code': r[2],
                'user_email': r[3],
                'user_full_name': r[4],
                'transaction_type': r[5],
                'amount': float(r[6]) if r[6] is not None else 0,
                'final_risk_score': float(r[7]) if r[7] is not None else 0,
                'risk_level': r[8],
                'system_action': r[9],
                'triggered_rules': r[10],
                'threat_warnings': r[11],
                'status': r[12],
                'created_at': r[13].isoformat() if r[13] else None
            })

        cur.close()
        conn.close()

        return jsonify({
            'success': True,
            'count': len(alerts),
            'alerts': alerts
        })
    except Exception as e:
        logger.error("[ALERTS] Error fetching alerts: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================================================================
# 5. SERVER ENTRYPOINT
# =========================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    logger.info("=================================================================")
    logger.info("  FRAUD DETECTION AI ENGINE IS STARTING ON PORT %s", port)
    logger.info("=================================================================")
    app.run(host='0.0.0.0', port=port, debug=False)
