#include <iostream>
using namespace std;

// Iterative Fibonacci function
int fibonacciIterative(int n)
{
    if (n == 0)
        return 0;
    if (n == 1)
        return 1;

    int a = 0, b = 1, fib = 1;
    for (int i = 2; i <= n; i++)
    {
        fib = a + b;
        a = b;
        b = fib;
    }
    return fib;
}

// Recursive Fibonacci function
int fibonacciRecursive(int n)
{
    if (n == 0)
        return 0;
    if (n == 1)
        return 1;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

// Iterative function to print Fibonacci series up to n
void printFibonacciSeriesIterative(int n)
{
    int a = 0, b = 1;
    cout << "Fibonacci Series (Iterative): ";
    for (int i = 0; i <= n; i++)
    {
        cout << a << " ";
        int next = a + b;
        a = b;
        b = next;
    }
    cout << endl;
}

// Recursive function to print Fibonacci series up to n
void printFibonacciSeriesRecursive(int a, int b, int n)
{
    if (n < 0)
        return;
    cout << a << " ";
    printFibonacciSeriesRecursive(b, a + b, n - 1);
}

int main()
{
    int n;
    cout << "Enter the number: ";
    cin >> n;

    // Calling both functions for individual nth Fibonacci term
    cout << "Fibonacci (Iterative) of " << n << " is: " << fibonacciIterative(n) << endl;
    cout << "Fibonacci (Recursive) of " << n << " is: " << fibonacciRecursive(n) << endl;

    // Calling functions to print the series up to nth term
    printFibonacciSeriesIterative(n);
    cout << "Fibonacci Series (Recursive): ";
    printFibonacciSeriesRecursive(0, 1, n);
    cout << endl;

    return 0;
}
