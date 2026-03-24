# Customer Return Prediction Project

##  Project Overview

This project demonstrates a **machine learning pipeline** for predicting whether customers will return for future purchases based on historical sales data. The dataset is **synthetic**, generated purely for **educational purposes**, and simulates sales transactions across products, regions, and customer types.

The workflow includes **data preparation, feature engineering, modeling, evaluation, and insights extraction**, making it suitable for learning real-world predictive analytics.

---

## 🗂 Dataset Description

The dataset simulates individual sales events and contains the following columns:

| Column | Description |
|--------|-------------|
| `Product_ID` | Unique identifier for each product sold |
| `Sale_Date` | Date of the sale (year 2023) |
| `Sales_Rep` | Sales representative handling the transaction (Alice, Bob, Charlie, David, Eve) |
| `Region` | Geographic region of sale (North, South, East, West) |
| `Sales_Amount` | Total sale amount (including discounts), range 100–10,000 |
| `Quantity_Sold` | Number of units sold per transaction, 1–50 |
| `Product_Category` | Category of product (Electronics, Furniture, Clothing, Food) |
| `Unit_Cost` | Cost per unit, range 50–5000 |
| `Unit_Price` | Selling price per unit (higher than cost) |
| `Customer_Type` | New or Returning customer |
| `Discount` | Applied discount (0–30%) |
| `Payment_Method` | Payment method (Credit Card, Cash, Bank Transfer) |
| `Sales_Channel` | Online or Retail |
| `Region_and_Sales_Rep` | Combined column for tracking purposes |

---

## 🛠 Tech Stack

* **Python 3.x** – Programming language  
* **Pandas & NumPy** – Data manipulation  
* **Matplotlib & Seaborn** – Data visualization  
* **Scikit-learn** – Machine learning, preprocessing, and evaluation  
* **Gradient Boosting Classifier** – Predictive model  

---


## 📊 Key Results

**Customer Future Return Distribution:**

| Status | Count | Percentage |
|--------|-------|------------|
| Not Returning | 206 | 55.98% |
| Returning | 162 | 44.02% |

**Model Performance:**

* **Accuracy:** 0.581  

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|--------|
| 0 (Not Returning) | 0.70 | 0.52 | 0.60 | 44 |
| 1 (Returning) | 0.49 | 0.67 | 0.56 | 30 |

**Top Features Influencing Customer Return:**

| Feature | Importance |
|---------|------------|
| `recency_days` | 0.197 |
| `total_profit` | 0.178 |
| `total_quantity` | 0.166 |
| `total_sales` | 0.155 |
| `avg_sales` | 0.146 |

> **Insight:** Customers who purchased more recently are more likely to return.

---

## 📈 Visualizations

* **Customer Return Distribution** 
<img width="640" height="480" alt="plot_2026-03-24 23-27-34_0" src="https://github.com/user-attachments/assets/439db795-f17d-4595-bc91-b2769d58260c" />

* **Customer Behavior** 
<img width="640" height="480" alt="plot_2026-03-24 23-27-34_1" src="https://github.com/user-attachments/assets/2544a05f-f50b-4888-a7e9-04c823339c4b" />

* **Confusion Matrix**
<img width="640" height="480" alt="plot_2026-03-24 23-27-34_2" src="https://github.com/user-attachments/assets/380f65cf-fdf4-4eef-a83e-7b40d7546296" />


* **ROC Curve**  
<img width="640" height="480" alt="plot_2026-03-24 23-27-34_3" src="https://github.com/user-attachments/assets/22739b3a-da76-4037-b0e5-adee1c3d4340" />

* **Feature Importance** – Bar chart of top drivers of return  
<img width="640" height="480" alt="plot_2026-03-24 23-27-34_4" src="https://github.com/user-attachments/assets/6dc1bebe-9c06-4d3d-95a6-a71e36dc71b0" />

---

## 💡 Insights & Recommendations

* **Recency Matters:** Recent customers are more likely to return.  
* **Profit & Quantity:** High-profit, high-quantity customers are key retention targets.  
* **Sales Strategy:** Targeted campaigns for profitable, recent customers can improve retention.  
* **Model Limitation:** Accuracy is moderate (~58%), suggesting scope for feature or model improvements.  



