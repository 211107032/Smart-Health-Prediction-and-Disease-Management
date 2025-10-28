#include<iostream>
using namespace std;

class furniture
{
    string material;
    float price;

    public:
          void accept()
          {
            cout << "\n Enter material & price of furniture:";
            cin >> material >> price;
          }
          void display()
          {
            cout << "\n Material of furniture is :"<< material;
            cout << "\n price of furniture is :"<< price;
          }
};
class table : public furniture
{
    int height ,surface;

    public:
         void accept1()
         {
            cout << "\n Enter height & surface of table:";
            cin >> height >> surface;
         }
         void display1()
         {
            cout << "Height of table :"<< height;
            cout << "surface of table :"<< surface;

         }
};
int main()
{
    table t;
    t.accept1();
    t.display();
    t.accept1();
    t.display1();

    return 0;
}