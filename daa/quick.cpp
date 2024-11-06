#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int deterministicPartition(vector<int> &arr, int low, int high)
{
    int pivot = arr[low];
    int i = low;
    int j = high;

    while (i < j)
    {
        while (arr[i] <= pivot && i <= high - 1)
            i++;
        while (arr[j] > pivot && j >= low + 1)
            j--;

        if (i < j)
            swap(arr[i], arr[j]);
    }

    swap(arr[low], arr[j]);
    return j;
}

void deterministicQuickSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pIndex = deterministicPartition(arr, low, high);
        deterministicQuickSort(arr, low, pIndex - 1);
        deterministicQuickSort(arr, pIndex + 1, high);
    }
}

int randomizedPartition(vector<int> &arr, int low, int high)
{
    int randomIndex = low + rand() % (high - low + 1);
    swap(arr[low], arr[randomIndex]);
    return deterministicPartition(arr, low, high);
}

void randomizedQuickSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pIndex = randomizedPartition(arr, low, high);
        randomizedQuickSort(arr, low, pIndex - 1);
        randomizedQuickSort(arr, pIndex + 1, high);
    }
}

vector<int> copyArray(const vector<int> &arr)
{
    return vector<int>(arr.begin(), arr.end());
}

int main()
{
    // Generate a random array of 10,000 elements
    // int n = 10000;
    // vector<int> arr(n);
    // for (int i = 0; i < n; ++i) {
    //     arr[i] = rand() % 10000; // Random numbers between 0 and 9999
    // }
    // Time taken by Deterministic QuickSort: 0.001333 seconds
    // Time taken by Randomized QuickSort: 0.001649 seconds

    int n = 10000;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i;
    }
    // Time taken by Deterministic QuickSort: 0.125452 seconds
    // Time taken by Randomized QuickSort: 0.000897 seconds

    // so if the array is already sorted then randomized works better
    //  than the deterministic quicksort

    // Copy the array for using it in both sorts
    vector<int> arr1 = copyArray(arr);
    vector<int> arr2 = copyArray(arr);

    auto start1 = chrono::high_resolution_clock::now();
    deterministicQuickSort(arr1, 0, n - 1);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> duration1 = end1 - start1;

    auto start2 = chrono::high_resolution_clock::now();
    randomizedQuickSort(arr2, 0, n - 1);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> duration2 = end2 - start2;

    cout << "Time taken by Deterministic QuickSort: " << duration1.count() << " seconds" << endl;
    cout << "Time taken by Randomized QuickSort: " << duration2.count() << " seconds" << endl;

    return 0;
}
