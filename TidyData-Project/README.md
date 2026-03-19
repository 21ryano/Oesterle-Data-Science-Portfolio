🧹 **Tidy Data Project – 2008 Olympic Medalists**

📊 *Project Overview*
This project transforms a messy dataset into a clean, structured format using tidy data principles. The dataset contains medalists from the 2008 Olympics, but its original format is difficult to analyze due to combined variables and missing values.


*The goal of this project is to:*
- Reshape the dataset into a tidy format
- Clean and standardize the data
- Perform exploratory data analysis (EDA)
- Visualize key insights about Olympic medal distribution

*📌 What is Tidy Data?*
- Tidy data is a standardized way of organizing datasets so they are easier to analyze. The three core principles are:
- Each variable has its own column
- Each observation has its own row
- Each type of observational unit forms its own table

Applying these principles makes data easier to manipulate, visualize, and model.


*⚙️ Instructions (How to Run the Project)*
1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Pandas, MatPlotLib, and Seaborn. This is intended for a Jupyter Notebook.
3. Run the Notebook


🗂️**Dataset Description**

The dataset (olympics_08_medalists.csv) contains information about athletes who won medals in the 2008 Olympics.

🔍 *Original Data Issues:*
- Columns combined multiple variables (e.g., gender + sport)
- Large number of missing values
- Inconsistent formatting



🛠️ **How I Fixed This:
- Used pd.melt() to reshape the dataset
- Split combined columns into gender and sport
- Removed null values
- Cleaned text formatting (spacing, capitalization, symbols)

✅ **Final Result:**
A tidy dataset with the following columns:
- medalist_name
- medal
- gender
- sport


💡 **Key Takeaways**
- Real-world data is often messy and requires significant cleaning
- Tidy data principles simplify analysis and visualization
- Python tools like pandas make reshaping data efficient
- Clean data enables better insights and storytelling


📚 *References*
📄 Hadley Wickham, Tidy Data Paper (2014):
https://vita.had.co.nz/papers/tidy-data.pdf
🧾 Tidy Data Cheat Sheet (RStudio):
https://rstudio.com/resources/cheatsheets/



📈 **Visual Examples**
![alt text](https://github.com/21ryano/Oesterle-Data-Science-Portfolio/blob/main/TidyData-Project/Screenshots/Screenshot%202026-03-19%20101020.png)
![alt text](https://github.com/21ryano/Oesterle-Data-Science-Portfolio/blob/main/TidyData-Project/Screenshots/Screenshot%202026-03-19%20101050.png)
