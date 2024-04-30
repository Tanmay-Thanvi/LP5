#include <bits/stdc++.h>
using namespace std;

template <typename T>
class Stats
{
private:
    int n;
    vector<T> vec;

public:
    Stats(int n);
    void Input();

    T Min();
    T Max();
    T Sum();
    double Avg();

    ~Stats();
};

template <typename T>
Stats<T>::Stats(int n)
{
    this->n = n;
    vec.resize(n);
}

template <typename T>
void Stats<T>::Input()
{
    cout << "Enter any (" << n << ") no.s : ";
    for (int i = 0; i < n; i++)
    {
        cin >> vec[i];
    }
}

template <typename T>
T Stats<T>::Min()
{
    T result = vec[0];
    #pragma omp parallel for reduction(min : result)
    for (int i = 1; i < n; i++)
    {
        result = min(result, vec[i]);
    }
    return result;
}

template <typename T>
T Stats<T>::Max()
{
    T result = vec[0];
    #pragma omp parallel for reduction(max : result)
    for (int i = 1; i < n; i++)
    {
        result = max(result, vec[i]);
    }
    return result;
}

template <typename T>
T Stats<T>::Sum()
{
    T result = vec[0];
    #pragma omp parallel for reduction(+ : result)
    for (int i = 1; i < n; i++)
    {
        result += vec[i];
    }
    return result;
}

template <typename T>
double Stats<T>::Avg()
{
    T result = Sum();
    double avg = (1.0 * result) / n;
    return avg;
}

template <typename T>
Stats<T>::~Stats()
{
    vec.clear();
}

int main()
{
    int n;

    cout<<"Enter no. of elements in array : ";
    cin >> n;
    Stats<int> s(n);
    
    s.Input();
    cout << "Min : " << s.Min() << endl;
    cout << "Max : " << s.Max() << endl;
    cout << "Sum : " << s.Sum() << endl;
    cout << "Avg : " << s.Avg() << endl;

    return 0;
}