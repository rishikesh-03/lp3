#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int knapsackRecursive(vector<int> &wt, vector<int> &val, int ind, int W)
{
    if (ind < 0)
    {
        return 0;
    }
    int notTaken = knapsackRecursive(wt, val, ind - 1, W);
    int taken = INT_MIN;
    if (wt[ind] <= W)
        taken = val[ind] + knapsackRecursive(wt, val, ind - 1, W - wt[ind]);
    return max(notTaken, taken);
}

int knapsackUtil(vector<int> &wt, vector<int> &val, int ind, int W, vector<vector<int>> &dp)
{
    if (ind < 0)
    {
        return 0;
    }
    if (dp[ind][W] != -1)
        return dp[ind][W];
    int notTaken = knapsackUtil(wt, val, ind - 1, W, dp);
    int taken = INT_MIN;
    if (wt[ind] <= W)
        taken = val[ind] + knapsackUtil(wt, val, ind - 1, W - wt[ind], dp);
    return dp[ind][W] = max(notTaken, taken);
}

int knapsackMemoization(int n, int W, vector<int> &val, vector<int> &wt)
{
    vector<vector<int>> dp(n, vector<int>(W + 1, -1));
    return knapsackUtil(wt, val, n - 1, W, dp);
}

int knapsackBottomUp(int n, int W, vector<int> &val, vector<int> &wt)
{
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    for (int ind = 1; ind <= n; ind++)
    {
        for (int cap = 0; cap <= W; cap++)
        {
            int notTaken = dp[ind - 1][cap];
            int taken = INT_MIN;
            if (wt[ind - 1] <= cap)
                taken = val[ind - 1] + dp[ind - 1][cap - wt[ind - 1]];
            dp[ind][cap] = max(notTaken, taken);
        }
    }
    return dp[n][W];
}

int main()
{
    int n = 20; // Number of items
    int W = 50; // Weight capacity of the knapsack

    // Generate random weights and values
    vector<int> wt(n);
    vector<int> val(n);
    srand(time(0)); // Seed for random number generation

    for (int i = 0; i < n; i++)
    {
        wt[i] = rand() % 10 + 1;   // Random weight between 1 and 10
        val[i] = rand() % 100 + 1; // Random value between 1 and 100
    }

    cout << "Weights: ";
    for (int weight : wt)
    {
        cout << weight << " ";
    }
    cout << endl;

    cout << "Values: ";
    for (int value : val)
    {
        cout << value << " ";
    }
    cout << endl;

    int method;
    while (true)
    {
        cout << "Choose method (1: Normal Recursion, 2: Bottom-Up, 3: Memoization, 0: Exit): ";
        cin >> method;

        if (method == 0)
        {
            break; // Exit the loop if the user chooses to exit
        }

        int maxValue = 0;

        switch (method)
        {
        case 1:
        {
            auto start = chrono::high_resolution_clock::now(); // Start timing for normal recursion
            maxValue = knapsackRecursive(wt, val, n - 1, W);
            auto end = chrono::high_resolution_clock::now(); // End timing
            chrono::duration<double> elapsed = end - start;  // Calculate elapsed time
            cout << "Time taken for Normal Recursion: " << elapsed.count() << " seconds" << endl;
            break;
        }
        case 2:
        {
            auto start = chrono::high_resolution_clock::now(); // Start timing for bottom-up DP
            maxValue = knapsackBottomUp(n, W, val, wt);
            auto end = chrono::high_resolution_clock::now(); // End timing
            chrono::duration<double> elapsed = end - start;  // Calculate elapsed time
            cout << "Time taken for Bottom-Up DP: " << elapsed.count() << " seconds" << endl;
            break;
        }
        case 3:
        {
            auto start = chrono::high_resolution_clock::now(); // Start timing for memoization
            maxValue = knapsackMemoization(n, W, val, wt);
            auto end = chrono::high_resolution_clock::now(); // End timing
            chrono::duration<double> elapsed = end - start;  // Calculate elapsed time
            cout << "Time taken for Memoization: " << elapsed.count() << " seconds" << endl;
            break;
        }
        default:
            cout << "Invalid method selected." << endl;
            continue; // Skip to the next iteration of the loop
        }

        // Output the maximum value the thief can steal
        cout << "The Maximum value of items the thief can steal is " << maxValue << endl;
    }

    cout << "Exiting the program." << endl;
    return 0; // Return 0 to indicate successful program execution
}