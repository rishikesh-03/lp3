#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Item
{
    double value, weight;
    Item(double v, double w) : value(v), weight(w) {}
};

bool compare(Item a, Item b)
{
    double r1 = a.value / a.weight;
    double r2 = b.value / b.weight;
    return r1 > r2;
}

double fractionalKnapsack(double W, vector<Item> &items)
{
    sort(items.begin(), items.end(), compare);

    double totalValue = 0.0;
    double currentWeight = 0;

    for (auto &item : items)
    {
        if (currentWeight + item.weight <= W)
        {
            // Take full item
            currentWeight += item.weight;
            totalValue += item.value;
        }
        else
        {
            double remainingWeight = W - currentWeight;
            totalValue +=  remainingWeight*(item.value  / item.weight);
            break;
        }
    }

    return totalValue;
}

int main()
{
    double W = 50;
    vector<Item> items = {{60, 10}, {100, 20}, {120, 30}};

    cout << "Maximum value in knapsack = " << fractionalKnapsack(W, items) << endl;
    return 0;
}
