#include <iostream> // добавляем функции для ввода и вывода на экран
#include <math.h> // добавляем математические функции
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
	double h = 0.1;// задаём шаг интегрирования

	double n; // задаём число разбиений n

	n = (b - a) / h;

	// вычисляем интеграл по формуле Симпсона
	Integral = h * (f(a) + f(b)) / 6.0;
	for (i = 1; i <= n; i++)
		Integral = Integral + 4.0 / 6.0 * h * f(a + h * (i - 0.5));
	for (i = 1; i <= n - 1; i++)
		Integral = Integral + 2.0 / 6.0 * h * f(a + h * i);
	cout << "I = " << Integral << "\n";

	system("pause");
}