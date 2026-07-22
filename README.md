# Macabets V6 — Local Tennis Database

Macabets now uses tennis files stored inside the GitHub repository. Streamlit no longer downloads the database when the page loads.

## First setup

1. Upload every file and folder in this package to the repository.
2. In GitHub, open **Actions**.
3. Select **Update Macabets Tennis Data**.
4. Click **Run workflow**.
5. Wait for the action to finish and commit the `data` files.
6. Reboot the Streamlit app.

The action also checks for updated tennis results once per day.

## Main result

Macabets runs 50,000 simulations and displays:

`Player A won 33,850 of 50,000 simulations — 67.7%`

It also displays the fair line, price comparison, recommendation, supporting factors and risks.

## Data source

ATP data by Jeff Sackmann / Tennis Abstract. CC BY-NC-SA 4.0. Attribution required; noncommercial use only.
