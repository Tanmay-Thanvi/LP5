#include <bits/stdc++.h>
using namespace std;

// Timer Class to note execution time

class Timer
{
private:
    chrono::high_resolution_clock::time_point startTime;
    chrono::high_resolution_clock::time_point endTime;
    long long elapsedTime;

public:
    Timer() {}

    void start()
    {
        startTime = chrono::high_resolution_clock::now();
    }

    void stop()
    {
        endTime = chrono::high_resolution_clock::now();
        elapsedTime = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime).count();
    }

    long long GetElapsedTime() const
    {
        return elapsedTime;
    }

    ~Timer()
    {
        cout << "Timer destroyed successfully!" << endl;
    }
};

// Algorithm Implementation Class

class Algorithm
{
private:
    int n;
    vector<int> v;

public:
    Algorithm(int n);
    ~Algorithm();

    // IO methods
    void Input();
    void printArr();

    // Sorting Algorithms
    Timer *BubbleSort();
    Timer *ParallelBubbleSort();
    Timer *MergeSort(int begun, int end);
    Timer *ParallelMergeSort(int begin, int end);
    void merge(int beg, int mid, int end);
};

// Member functions for the algorithm

Algorithm::Algorithm(int n)
{
    this->n = n;
    v.resize(n);
}

void Algorithm::Input()
{
    cout << "Enter (" << n << ") elements : " << endl;
    for (int i = 0; i < n; i++)
    {
        cin >> v[i];
    }
}

void Algorithm::printArr()
{
    cout << "Arr : ";
    for (int i = 0; i < n; i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

Timer *Algorithm::BubbleSort()
{
    Timer *t;
    t = new Timer();
    t->start();

    bool swapped;

    for (int i = 0; i < n; i++)
    {
        swapped = false;
        for (int j = i; j < n; j++)
        {
            if (v[j] < v[i])
            {
                swap(v[j], v[i]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }

    t->stop();

    return t;
}

Timer *Algorithm::ParallelBubbleSort()
{
    Timer *t;
    t = new Timer();
    t->start();

    bool swapped;

    for (int i = 0; i < n; i++)
    {
        swapped = false;

        #pragma omp parallel for shared(v, swapped)
        for (int j = i; j < n; j++)
        {
            if (v[j] < v[i])
            {
                swap(v[j], v[i]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }

    t->stop();

    return t;
}

void Algorithm::merge(int beg, int mid, int end)
{
    int i, j, k;
    int n1 = mid - beg + 1;
    int n2 = end - mid;

    int LeftArray[n1], RightArray[n2]; // temporary arrays

    /* copy data to temp arrays */
    for (int i = 0; i < n1; i++)
        LeftArray[i] = v[beg + i];

    for (int j = 0; j < n2; j++)
        RightArray[j] = v[mid + 1 + j];

    i = 0;   /* initial index of first sub-array */
    j = 0;   /* initial index of second sub-array */
    k = beg; /* initial index of merged sub-array */

    while (i < n1 && j < n2)
    {
        if (LeftArray[i] <= RightArray[j])
        {
            v[k] = LeftArray[i];
            i++;
        }
        else
        {
            v[k] = RightArray[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        v[k] = LeftArray[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        v[k] = RightArray[j];
        j++;
        k++;
    }
}

Timer *Algorithm::MergeSort(int begin, int end)
{
    Timer *t;
    t = new Timer();
    t->start();

    if (begin >= end)
    {
        t->stop();
        return t;
    }

    int mid = begin + (end - begin) / 2;

    // Sort first and second halves
    MergeSort(0, mid);
    MergeSort(mid + 1, end);

    // Merge the sorted halves
    merge(begin, mid, end);

    t->stop();

    return t;
}

Timer *Algorithm::ParallelMergeSort(int begin, int end)
{
    Timer *t;
    t = new Timer();
    t->start();

    if (begin >= end)
    {
        t->stop();
        return t;
    }

    int mid = begin + (end - begin) / 2;

    // Sort first and second halves
    #pragma omp parallel sections
    {
        #pragma omp section
            MergeSort(0, mid);
        #pragma omp section
            MergeSort(mid + 1, end);
    }

    // Merge the sorted halves
    merge(begin, mid, end);

    t->stop();

    return t;
}

Algorithm::~Algorithm()
{
    n = 0;
    v.clear();
}

// Main Function
int main()
{
    int n;
    cout << "Enter the size of the array : ";
    cin >> n;

    Algorithm a(n);
    a.Input();
    cout << endl;

    // Menu
    bool choice1 = false, choice2 = false, choice3 = false, choice4 = false;

    cout << "Enter your choice : " << endl;
    cout << "1. Bubble Sort" << endl;
    cout << "2. Parallel Bubble Sort" << endl;
    cout << "3. Merge Sort" << endl;
    cout << "4. Parallel Merge Sort" << endl;
    cout << "Choose (4 opts) : ";
    cin >> choice1 >> choice2 >> choice3 >> choice4;
    cout << endl;

    if (choice1)
    {
        // -------------- Bubble Sort -------------- //
        Timer *t1 = a.BubbleSort();
        cout << "Sorted Array : ";
        a.printArr();
        cout << "Time taken by Bubble Sort: " << t1->GetElapsedTime() << " nano seconds \n"
             << endl;

        // delete Timer 
        delete(t1);
    }

    if (choice2)
    {
        // -------- Parallel Bubble Sort ----------- //
        Timer *t2 = a.ParallelBubbleSort();
        cout << "Sorted Array : ";
        a.printArr();
        cout << "Time taken by Parallel Bubble Sort: " << t2->GetElapsedTime() << " nano seconds \n"
             << endl;

        // delete Timer 
        delete(t2);
    }

    if (choice3)
    {
        // ------------- Merge Sort --------------- //
        Timer *t3 = a.MergeSort(0, n - 1);
        cout << "Sorted Array : ";
        a.printArr();
        cout << "Time taken by Merge Sort: " << t3->GetElapsedTime() << " nano seconds \n"
             << endl;

        // delete Timer 
        delete(t3);
    }
    if (choice4)
    {
        // ---------- Parallel Merge Sort --------- //
        Timer *t4 = a.ParallelMergeSort(0, n - 1);
        cout << "Sorted Array : ";
        a.printArr();
        cout << "Time taken by Parallel Merge Sort: " << t4->GetElapsedTime() << " nano seconds \n"
             << endl;

        // delete Timer 
        delete(t4);
    }

    return 0;
}