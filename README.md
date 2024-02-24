# Software Reliability using CGWO + SVM

# Demo
  You can watch the demo for the above project on YouTube here:
  `https://youtu.be/9-b5Hy_B02g?si=w9bSCFa-LXjmgeMX`

# About CGWO
  Chaotic Grey Wolf Optimization, abbreviated as CGWO, is a bio-inspired feature selection algorithm. 
  It selects a subset of most useful features by replicating the techinque of hunting used by Grey Wolves in nature.
  Algorithm:
  ```
  1. Search agents(wolves) are initialized.
  2. Alpha, Beta and Delta wolves are chosen.
  3. The rest of the pack modify their vectors inorder to follow the alpha wolf.
  4. Steps 2,3 are repeated for a certain number of iterations till the search space is exhausted. 
  ```
# How to run the app

+ Clone the repository
  ```
    git clone https://github.com/pranav-salunkhe/Google-Solution-Challenge.git
  ```

+ Add .env file with the following contents:
  ```
  GEMINI_API=your_api_key
  TORCH_API=your_api_key
  ```
+ Run the app:
  ```
  python3 app.py
  ```

# Dataset
  Nasa JM1 Dataset

# Tech Stack
  + Gemini API
  + Flask
  + TailwindCSS
  + Python (sklearn, numpy, pandas)

  
