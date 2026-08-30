**Overview**  
The dataset contains 156 customer records and 8 variables that describe each customer’s demographic profile, subscription history, and purchasing behaviour. While the overall size is modest, a handful of data‑quality gaps—duplicates, missing entries, inconsistent formatting, and a few extreme values—will need to be addressed before the data can support reliable analysis or predictive modeling.

**Data Quality Issues Found**  
A small number of duplicate rows (6) should be removed to avoid double‑counting customers. Several columns have missing values: age (12 rows), region (13 rows), and monthly spend (12 rows). The age column also includes a non‑numeric placeholder (“unknown”), which must be converted to a proper numeric format or imputed. In the region and is_active columns, capitalisation inconsistencies (“North” vs “north”) could skew grouping operations, so standardising the text is recommended. Finally, three records have monthly spend values that stand out from the rest; flagging or investigating these outliers will help ensure they do not distort statistical tests.

**Key Findings**  
Only one of the five hypotheses we tested reached statistical significance. The comparison of average age between active and inactive customers produced a significant p‑value (p ≈ 0.02), indicating that the two groups differ meaningfully in age. This suggests that age is a factor in a customer’s likelihood of remaining active, which could inform targeted retention strategies. The remaining hypotheses—correlation between age and spend, regional spend differences, spend over time, and spend by active status—did not show statistically significant differences (all p‑values well above the conventional 0.05 threshold). In practice, this means that, based on the current data, we cannot conclude that age drives spending, that spending patterns differ by region or over time, or that active customers spend more than inactive ones.

**Visualizations**  
Four charts are available to help communicate the results visually:  

* **Age vs Monthly Spend Scatter** – plots each customer’s age against their monthly spend, allowing us to see whether a linear relationship exists; the plot confirms the very weak trend identified by the correlation test.  
* **Monthly Spend by Region Bar** – a bar chart showing the average spend per region; the bars are nearly identical, mirroring the non‑significant ANOVA result.  
* **Monthly Spend by Active Status Box** – a boxplot that contrasts spend distributions for active versus inactive customers; the boxes overlap almost completely, again supporting the lack of a significant difference.  
* **Age by Active Status Box** – a boxplot that visually highlights the age disparity between the two groups, reinforcing the significant result from the ANOVA test.

**Readiness for Modeling**  
Before the data can be fed into a machine‑learning model, several feature‑engineering steps are required:  

1. **Remove High‑Cardinality Identifiers** – columns such as name, signup_date, email, and customer_id contain unique values that provide no predictive power; dropping them reduces noise.  
2. **Scale Numeric Features** – age and monthly spend should be standardised (e.g., z‑score scaling) so that models sensitive to feature scale converge more efficiently.  
3. **Encode Categorical Variables** – convert the binary is_active column to 0/1 labels, and transform the low‑cardinality region column into one‑hot vectors to preserve categorical distinctions.  

In addition to these transformations, the earlier cleaning actions (duplicate removal, imputation of missing values, dtype correction, category standardisation, and outlier flagging) must be completed. Once these steps are executed, the dataset will be in a clean, model‑ready state, enabling more robust predictive analytics and deeper insights into customer behaviour.