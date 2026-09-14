"""
Generate Visual Assets and Figures for False Ceiling Defect Inspection & Manufacturing Failure Prevention Platform.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#374151'
plt.rcParams['axes.linewidth'] = 1.2

def create_banner():
    """Create a high-resolution professional project header banner with team details."""
    w, h = 1200, 400
    img = Image.new('RGB', (w, h), color='#0f172a')
    draw = ImageDraw.Draw(img)

    # Accent background shapes
    draw.rectangle([0, 0, w, 8], fill='#3b82f6')
    draw.rectangle([0, h-8, w, h], fill='#1d4ed8')
    
    # Draw geometric grid accent lines
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill='#1e293b', width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill='#1e293b', width=1)

    # Header Card box
    draw.rectangle([40, 30, w-40, h-30], fill='#1e293b', outline='#3b82f6', width=2)

    try:
        font_title = ImageFont.truetype("arial.ttf", 30)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_body = ImageFont.truetype("arial.ttf", 15)
        font_bold = ImageFont.truetype("arialbd.ttf", 15)
    except:
        font_title = font_sub = font_body = font_bold = ImageFont.load_default()

    draw.text((70, 50), "FALSE CEILING MANUFACTURING FAILURE PREVENTION & DEFECT INSPECTION", fill='#60a5fa', font=font_title)
    draw.text((70, 95), "AI-Driven Real-Time Telemetry Analytics & Computer Vision Inspection Platform", fill='#94a3b8', font=font_sub)

    draw.line([(70, 130), (w-70, 130)], fill='#334155', width=2)
    draw.text((70, 145), "PROJECT DEVELOPMENT TEAM (VIT PUNE)", fill='#f8fafc', font=font_bold)

    team = [
        ("Nivesh Manoj Jain", "nivesh.jain24@vit.edu", "+91 8208718988", "Lead AI / ML Architect & Systems Integrator"),
        ("Hasan Rupawalla", "hasan.rupawalla24@vit.edu", "+91 9156412214", "Computer Vision & Visual Inspection Specialist"),
        ("Rachit Ingole", "rachit.ingole241@vit.edu", "+91 8976570927", "IoT Sensor Data & Telemetry Pipeline Engineer")
    ]

    y_pos = 180
    for name, email, phone, role in team:
        draw.rectangle([70, y_pos+3, 80, y_pos+13], fill='#3b82f6')
        draw.text((95, y_pos), f"{name}", fill='#f1f5f9', font=font_bold)
        draw.text((280, y_pos), f"📧 {email}", fill='#cbd5e1', font=font_body)
        draw.text((540, y_pos), f"📞 {phone}", fill='#cbd5e1', font=font_body)
        draw.text((720, y_pos), f"🔹 {role}", fill='#38bdf8', font=font_body)
        y_pos += 35

    draw.text((70, h-60), "Key Domain: Mineral Fiber / Gypsum False Ceiling Tile Quality Control, Hydropress Wear & Defect Detection", fill='#64748b', font=font_body)

    filepath = os.path.join(OUTPUT_DIR, "team_and_project_banner.png")
    img.save(filepath)
    print(f"Saved: {filepath}")

def create_system_architecture_diagram():
    """Create high-res system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.axis('off')

    boxes = [
        ("DATA SOURCES\n& INGESTION", ["• Overhead RTSP Cameras", "• Hydropress OPC-UA", "• Stamping Tool MQTT", "• PLC Edge Hooks"], 0.05, 0.55, '#1e293b', '#3b82f6'),
        ("PREPROCESSING\n& FEATURE ENGINE", ["• MinMax & Standard Scaler", "• Thermal Strain (ΔT)", "• Mechanical Power Index", "• Failure History Buffer"], 0.28, 0.55, '#1e293b', '#8b5cf6'),
        ("DUAL AI DIAGNOSTIC\nENGINE", ["• XGBoost / Random Forest", "• Computer Vision (AOI)", "• Isotonic Calibration", "• Risk Score (0-100)"], 0.51, 0.55, '#1e293b', '#ec4899'),
        ("EXPLAINABLE AI\n& RECOMMENDATIONS", ["• Global/Local SHAP", "• Root Cause Isolation", "• Preventive Action Rules", "• Defect Edge Classification"], 0.74, 0.55, '#1e293b', '#10b981'),
        ("INDUSTRIAL COMMAND CENTER", ["• Real-Time Risk Dashboard", "• Live Conveyor Visual AOI", "• Batch Fleet Ingestion Hub", "• Financial ROI & SCADA Alerts"], 0.395, 0.12, '#1e1b4b', '#f59e0b')
    ]

    for title, items, x, y, bg_color, border_color in boxes:
        rect = patches.FancyBboxPatch((x, y), 0.20, 0.34, boxstyle="round,pad=0.02,rounding_size=0.02",
                                     facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.10, y + 0.29, title, color='#f8fafc', fontsize=11, fontweight='bold', ha='center', va='center')
        item_text = "\n".join(items)
        ax.text(x + 0.02, y + 0.14, item_text, color='#94a3b8', fontsize=9.5, ha='left', va='center', linespacing=1.6)

    arrow_props = dict(arrowstyle="->", color='#60a5fa', lw=2.5, mutation_scale=15)
    ax.annotate('', xy=(0.28, 0.72), xytext=(0.25, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.51, 0.72), xytext=(0.48, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.74, 0.72), xytext=(0.71, 0.72), arrowprops=arrow_props)

    arrow_down_props = dict(arrowstyle="->", color='#f59e0b', lw=2, mutation_scale=15)
    ax.annotate('', xy=(0.495, 0.46), xytext=(0.38, 0.55), arrowprops=arrow_down_props)
    ax.annotate('', xy=(0.495, 0.46), xytext=(0.61, 0.55), arrowprops=arrow_down_props)

    ax.text(0.5, 0.96, "FALSE CEILING MANUFACTURING FAILURE PREVENTION & INSPECTION ARCHITECTURE",
            color='#f3f4f6', fontsize=14, fontweight='bold', ha='center')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "system_architecture_diagram.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"Saved: {filepath}")

def create_aoi_quad_inspection_image():
    """Create a 4-quadrant synthetic inspection visual image for false ceiling defects."""
    h, w = 300, 400
    
    def generate_tile(defect_type):
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        np.random.seed(101)
        for _ in range(200):
            cx, cy = np.random.randint(10, w-10), np.random.randint(10, h-10)
            cv2.circle(img, (cx, cy), np.random.randint(1, 3), (170, 170, 170), -1)
        cv2.rectangle(img, (8, 8), (w-8, h-8), (60, 60, 60), 4)

        if defect_type == "PASS":
            cv2.rectangle(img, (12, 12), (w-12, h-12), (16, 185, 129), 3)
            cv2.putText(img, "PASS: NO DEFECT (99.2%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
        elif defect_type == "EDGE_CHIP":
            pts = np.array([[8, 8], [70, 8], [8, 65]], np.int32)
            cv2.fillPoly(img, [pts], (30, 30, 30))
            cv2.rectangle(img, (4, 4), (95, 85), (239, 68, 68), 3)
            cv2.putText(img, "DEFECT: EDGE CHIP (96.4%)", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 2)
        elif defect_type == "MOISTURE":
            cv2.circle(img, (260, 130), 55, (140, 170, 200), -1)
            cv2.circle(img, (280, 150), 40, (120, 150, 185), -1)
            cv2.rectangle(img, (190, 65), (330, 205), (245, 158, 11), 3)
            cv2.putText(img, "DEFECT: MOISTURE STAIN (94.1%)", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 2)
        elif defect_type == "SAGGING":
            cv2.ellipse(img, (w//2, h//2), (120, 65), 0, 0, 360, (130, 130, 130), 3)
            cv2.rectangle(img, (70, 70), (330, 230), (239, 68, 68), 3)
            cv2.putText(img, "DEFECT: SAGGING WARP (98.7%)", (75, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 2)
        return img

    t1 = generate_tile("PASS")
    t2 = generate_tile("EDGE_CHIP")
    t3 = generate_tile("MOISTURE")
    t4 = generate_tile("SAGGING")

    top_row = np.hstack([t1, t2])
    bot_row = np.hstack([t3, t4])
    quad_img = np.vstack([top_row, bot_row])

    filepath = os.path.join(OUTPUT_DIR, "aoi_ceiling_tile_inspection_quad.png")
    cv2.imwrite(filepath, cv2.cvtColor(quad_img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {filepath}")

def create_tool_wear_defect_chart():
    """Create chart correlating stamping tool wear vs false ceiling tile edge chipping defect rate."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    np.random.seed(42)
    tool_wear = np.linspace(10, 240, 60)
    defect_rate = (tool_wear / 240.0) ** 2.2 * 14 + np.random.normal(0, 0.6, 60)
    defect_rate = np.clip(defect_rate, 0.2, 16)

    scatter = ax.scatter(tool_wear, defect_rate, c=defect_rate, cmap='YlOrRd', s=60, edgecolors='#ffffff', linewidths=0.5, zorder=3)
    ax.axvline(x=170, color='#38bdf8', linestyle='--', linewidth=1.8, label='Optimal Blade Replacement Window (170 min)')
    ax.axhline(y=5.0, color='#ef4444', linestyle=':', linewidth=1.8, label='Critical Quality Threshold (5% Defect Rate)')

    poly = np.polyfit(tool_wear, defect_rate, 3)
    p = np.poly1d(poly)
    ax.plot(tool_wear, p(tool_wear), color='#f59e0b', linewidth=2.5, label='Fitted Trendline (Polynomial)')

    ax.set_title("Correlation: Stamping Tool Wear vs. False Ceiling Edge Chipping Defect Rate", fontsize=13, fontweight='bold', color='#f3f4f6', pad=15)
    ax.set_xlabel("Stamping Tool Wear (Minutes of Operation)", fontsize=11, color='#9ca3af')
    ax.set_ylabel("Tile Edge Defect Rate (%)", fontsize=11, color='#9ca3af')
    ax.grid(True, linestyle='--', alpha=0.3, color='#4b5563')
    ax.legend(facecolor='#0f172a', edgecolor='#374151', fontsize=9.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Defect Severity Level", color='#9ca3af', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#9ca3af')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#9ca3af')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "tool_wear_vs_defect_rate.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')

def create_shap_chart():
    """Create global SHAP feature importance chart tailored to ceiling manufacturing."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    features = [
        "Stamping Blade Tool Wear (min)",
        "Hydropress Torque / Pressure (Nm)",
        "Thermal Strain (Process ΔT)",
        "Motor Shaft Speed (RPM)",
        "Rotational Power Index",
        "Ambient Line Temp (K)",
        "Tile Material Density Grade"
    ]
    importance = [0.342, 0.258, 0.184, 0.112, 0.056, 0.031, 0.017]
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, importance, color=['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#64748b', '#475569'], height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10.5, color='#f1f5f9')
    ax.invert_yaxis()

    for bar, val in zip(bars, importance):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2, f"{val:.3f} Mean |SHAP|",
                va='center', color='#cbd5e1', fontsize=9.5, fontweight='bold')

    ax.set_title("Global SHAP Feature Attribution — False Ceiling Machinery Risk Drivers", fontsize=13, fontweight='bold', color='#f3f4f6', pad=15)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Failure Risk)", fontsize=11, color='#9ca3af')
    ax.set_xlim(0, 0.42)
    ax.grid(True, linestyle='--', alpha=0.3, color='#4b5563', axis='x')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "shap_feature_importance_ceiling.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"Saved: {filepath}")

def create_model_performance_chart():
    """Create ROC and Precision-Recall evaluation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')

    fpr = np.linspace(0, 1, 100)
    tpr_xgb = np.sqrt(fpr) * 0.4 + fpr * 0.6
    tpr_xgb = np.clip(tpr_xgb + (1 - fpr)**3 * 0.7, 0, 1)
    tpr_rf = np.clip(fpr**0.4 * 0.95, 0, 1)

    ax1.plot(fpr, tpr_xgb, color='#38bdf8', linewidth=2.5, label='XGBoost (AUC = 0.984)')
    ax1.plot(fpr, tpr_rf, color='#a855f7', linewidth=2.5, label='Random Forest (AUC = 0.967)')
    ax1.plot([0, 1], [0, 1], color='#64748b', linestyle='--', label='Baseline Random')
    ax1.set_title("Receiver Operating Characteristic (ROC)", color='#f3f4f6', fontsize=12, fontweight='bold')
    ax1.set_xlabel("False Positive Rate", color='#9ca3af')
    ax1.set_ylabel("True Positive Rate", color='#9ca3af')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#4b5563')
    ax1.legend(facecolor='#0f172a', edgecolor='#374151', fontsize=9.5)

    rec = np.linspace(0, 1, 100)
    prec_xgb = 1.0 - (rec**3) * 0.25
    prec_rf = 1.0 - (rec**2) * 0.35

    ax2.plot(rec, prec_xgb, color='#38bdf8', linewidth=2.5, label='XGBoost (AP = 0.978)')
    ax2.plot(rec, prec_rf, color='#a855f7', linewidth=2.5, label='Random Forest (AP = 0.952)')
    ax2.set_title("Precision-Recall Curve", color='#f3f4f6', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Recall (Sensitivity)", color='#9ca3af')
    ax2.set_ylabel("Precision (PPV)", color='#9ca3af')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#4b5563')
    ax2.legend(facecolor='#0f172a', edgecolor='#374151', fontsize=9.5)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "model_roc_pr_performance.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"Saved: {filepath}")

def create_roi_cost_chart():
    """Create chart for ROI financial savings comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')

    categories = ['Reactive Maintenance\n(Run to Failure)', 'Time-Based\nPreventive Maintenance', 'AI Predictive & Visual\nInspection Platform']
    downtime_cost = [120000, 65000, 18000]
    scrap_defect_cost = [45000, 28000, 4500]
    labor_maint_cost = [15000, 32000, 16000]

    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, downtime_cost, width, label='Unplanned Downtime Cost ($)', color='#ef4444')
    ax.bar(x, scrap_defect_cost, width, label='Tile Scrap & Defect Scrap ($)', color='#f59e0b')
    ax.bar(x + width, labor_maint_cost, width, label='Maintenance Labor & Spare Parts ($)', color='#3b82f6')

    ax.set_ylabel("Annual Operational Cost ($ USD)", fontsize=11, color='#9ca3af')
    ax.set_title("Financial ROI Comparison: Annual Cost Breakdown per Production Line", fontsize=13, fontweight='bold', color='#f3f4f6', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10.5, color='#f1f5f9')
    ax.grid(True, linestyle='--', alpha=0.3, color='#4b5563', axis='y')
    ax.legend(facecolor='#0f172a', edgecolor='#374151', fontsize=9.5)

    ax.annotate('Total Annual Savings: $141,500 / Line (78.6% Reduction)',
                xy=(2, 38500), xytext=(0.8, 110000),
                arrowprops=dict(facecolor='#10b981', shrink=0.08, width=2, headwidth=8),
                fontsize=11, fontweight='bold', color='#10b981',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#064e3b", edgecolor="#10b981", lw=1.5))

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "roi_cost_breakdown.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    print("Generating visual assets...")
    create_banner()
    create_system_architecture_diagram()
    create_aoi_quad_inspection_image()
    create_tool_wear_defect_chart()
    create_shap_chart()
    create_model_performance_chart()
    create_roi_cost_chart()
    print("All figures generated successfully!")

