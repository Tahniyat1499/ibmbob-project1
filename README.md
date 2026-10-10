# 🛒 E-Commerce Analytics and Management System

**Author:** Tahniyat  
**Course Project | Data Analytics & Machine Learning**

---

## 📌 Project Description

A full-stack **E-Commerce Analytics and Management System** built on a real-world UK online retail dataset containing **541,910 transactions** spanning December 2010 – December 2011.

The system delivers:
- End-to-end **data cleaning and preprocessing**
- **Exploratory Data Analysis (EDA)** — revenue trends, sales patterns, geographic insights
- **RFM Customer Segmentation** (Recency, Frequency, Monetary)
- **Cohort Analysis** — customer retention heatmap
- **K-Means Clustering** — ML-based customer grouping
- **REST API Backend** built with Flask
- **Interactive Frontend Dashboard** (HTML/CSS/JavaScript + Chart.js) served via Flask

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `Tahniyat_EcommerceAnalytics.ipynb` | Main Jupyter Notebook — all analytics, API & UI |
| `requirements.txt` | Python dependencies |
| `Tahniyat_ProjectReport.docx` | Full project documentation report |
| `README.md` | This file |
| `data.csv` | Source dataset |

---

## 📊 Dataset

**Name:** Online Retail Dataset  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/ml/datasets/online+retail  
**Original Author:** Dr. Daqing Chen, London South Bank University

### Columns

| Column | Description |
|--------|-------------|
| `InvoiceNo` | Unique invoice number; prefix `C` = cancellation |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units per transaction |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit (£ GBP) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer country |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Jupyter Notebook                    │
│                                                      │
│  ┌──────────────┐   ┌──────────────────────────┐    │
│  │  Data Layer  │   │     Analytics Layer       │    │
│  │  data.csv    │──▶│  EDA · RFM · Cohort ·    │    │
│  │  pandas/numpy│   │  K-Means Clustering       │    │
│  └──────────────┘   └────────────┬─────────────┘    │
│                                  │                   │
│                    ┌─────────────▼─────────────┐    │
│                    │    Flask Backend API       │    │
│                    │    localhost:5000/api/*    │    │
│                    └─────────────┬─────────────┘    │
│                                  │  JSON             │
│                    ┌─────────────▼─────────────┐    │
│                    │  Frontend Dashboard (UI)   │    │
│                    │  localhost:5000/           │    │
│                    │  HTML · CSS · Chart.js     │    │
│                    └───────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 🔌 Backend API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend HTML dashboard |
| `GET` | `/api/summary` | KPI metrics (revenue, orders, customers, etc.) |
| `GET` | `/api/monthly-revenue` | Monthly revenue trend data |
| `GET` | `/api/top-products` | Top 10 products by revenue |
| `GET` | `/api/top-countries` | Top 10 countries by revenue |
| `GET` | `/api/rfm-segments` | Customer segment distribution |
| `GET` | `/api/revenue-by-dow` | Revenue by day of week |
| `GET` | `/api/revenue-by-hour` | Revenue by hour of day |

**Example request:**
```bash
curl http://localhost:5000/api/summary
```

**Example response:**
```json
{
  "total_revenue": 8905431.56,
  "total_orders": 22190,
  "total_customers": 4338,
  "total_products": 3684,
  "total_countries": 37,
  "avg_order_value": 401.33
}
```

---

## 🖥️ Frontend UI

The dashboard is served at `http://localhost:5000/` and includes:

- **KPI Cards** — Revenue, Orders, Customers, Products, Countries, Avg Order Value
- **Line Chart** — Monthly Revenue Trend
- **Bar Charts** — Revenue by Day of Week, Revenue by Hour
- **Horizontal Bar Charts** — Top 10 Products, Top 10 Countries
- **Doughnut Chart** — RFM Customer Segment Distribution
- **Data Tables** — RFM Segment Summary, Top Products

---

## 🛠️ Technologies Used

### Data & Analytics
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| pandas | ≥1.5.0 | Data manipulation |
| numpy | ≥1.23.0 | Numerical operations |
| matplotlib | ≥3.6.0 | Static visualisations |
| seaborn | ≥0.12.0 | Statistical plots |
| plotly | ≥5.13.0 | Interactive visualisations |
| scikit-learn | ≥1.2.0 | K-Means, StandardScaler, Silhouette Score |

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| Flask | ≥2.3.0 | REST API server |
| flask-cors | ≥4.0.0 | Cross-origin resource sharing |
| requests | ≥2.28.0 | API testing within notebook |

### Frontend
| Technology | Purpose |
|-----------|---------|
| HTML5 / CSS3 | Responsive layout, card components |
| JavaScript (ES2020) | Fetch API, async data loading |
| Chart.js v4 | Line, Bar, Doughnut charts |

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites
- Python 3.10 or higher
- pip or conda

### 2. Clone / Download the project
```bash
# If using git
git clone <your-repo-url>
cd "ecommerce analytics and management system"
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure the dataset is present
Make sure `data.csv` is in the same directory as the notebook.

### 5. Launch Jupyter Notebook
```bash
jupyter notebook Tahniyat_EcommerceAnalytics.ipynb
```

### 6. Run all cells
- Use **Kernel → Restart & Run All** to execute the entire notebook.
- The Flask server will start automatically in the last section.

### 7. Open the Dashboard
Navigate to **http://localhost:5000/** in your browser.

### 8. Test the API
```bash
# Summary KPIs
curl http://localhost:5000/api/summary

# Monthly revenue
curl http://localhost:5000/api/monthly-revenue

# Top products
curl http://localhost:5000/api/top-products
```

---

## 📈 Key Findings

1. **Revenue peaks in Q4** — November is the highest-revenue month (pre-Christmas).
2. **UK dominates** — ~85% of all revenue originates from the United Kingdom.
3. **B2B buyer behaviour** — Peak activity on Thursdays between 10am–2pm.
4. **Champions drive revenue** — A small segment of Champions generates the majority of sales.
5. **High Month-1 churn** — Retention drops to ~25–35% after the first purchase.
6. **Top product category** — Decorative home items (candle holders, seasonal décor, storage).

---

## 📁 Directory Structure

```
ecommerce analytics and management system/
│
├── Tahniyat_EcommerceAnalytics.ipynb   ← Main notebook
├── requirements.txt                     ← Python dependencies
├── Tahniyat_ProjectReport.docx          ← Project report
├── README.md                            ← This file
├── data.csv                             ← Dataset
└── data - Copy.csv                      ← Dataset backup
```

---

## 📜 License

This project is submitted for academic purposes. Dataset credit: Dr. Daqing Chen, UCI Machine Learning Repository.
