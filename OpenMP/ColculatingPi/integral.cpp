#include <iostream> // добавляем функции для ввода и вывода на экран
#include <math.h> // добавляем математические функции
#include <omp.h>
#include <stdio.h>
using namespace std;

// функция, интеграл
double f(double x)
{
    return 4.0 / (1.0 + x * x);
}

int main()
{
    int i; // счётчик
    double Integral; // здесь будет интеграл
    double a = 0.0, b = 1.0; // задаём отрезок интегрирования
    double h = 0.01;// задаём шаг интегрирования

    double n; // задаём число разбиений n

    n = (b - a) / h;

    // вычисляем интеграл по формуле трапеций

    Integral = h * (f(a) + f(b)) / 6.0;


    double Integral1 = 0;


#pragma omp parallel for shared(a,b,h,n) reduction(+:Integral1)
    for (int i = 1; i <= (int)n; i++)
    {
        Integral1 = Integral1 + h * f(a + h * i);
    }

    Integral = Integral + Integral1;

    cout << "I = " << Integral << "\n";
}


