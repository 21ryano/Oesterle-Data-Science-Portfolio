 # Tidy Data Project – 2008 Olympic Medalists

This project takes a messy dataset of 2008 Olympics medalists and transforms it into a clean, structured format using tidy data principles. The goal was to Reshape the dataset, Clean and standardize the data, Perform exploratory data analysis (EDA), Visualize key insights about Olympic medal distribution. 

**How to Use**
1. Copy the Repository
2. Install Pandas, MatPlotLib, and Seaborn. This is intended for a Jupyter Notebook.
3. Run the Notebook

**Dataset Description**
- The dataset (olympics_08_medalists.csv) contains information about athletes who won medals in the 2008 Olympics.
- *Original Data Issues:*
     - Columns combined multiple variables (e.g., gender + sport)
     - Large number of missing values
     - Inconsistent formatting
- *How I Fixed This:*
    -  Used pd.melt() to reshape the dataset
    -  Split combined columns into gender and sport
    -  Removed null values
    -  Cleaned text formatting (spacing, capitalization, symbols)

- ***Final Result:***

   - A tidy dataset with the following columns:
       - Medalist_Name
       - Medal
       - Gender
       - Sport
- **Each variable is its own column, each observation its own row, and each observation unit its own table**

**Key Takeaways**
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
