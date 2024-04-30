To run first 3 files : <br>
g++ -o exec_name -fopenmp file_name.cpp <br>
// where exec_name is the name of executable file & file_name is the code file

```cpp
// Parallelized OpenMp Code 
#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        std::cout << "Hello from thread " << id << std::endl;
    }
    return 0;
}
```

Note : You cannot run this code using coderunner. You need to use the command <br>
cmd : g++ -o exec_name -fopenmp file_name.cpp 

To Run CUDA assignment : <br>
commands are 
```bash
nvcc filename.cu -o execname
./execname
```