#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <math.h>

class Animal{
private:
    std::string name;
    double height;
    double weight;

    static int numOfAnimals;

public:
    std::string GetName(){return name;}
    void SetName(std::string name){this->name = name;}
    double GetHeight(){return height;}
    void SetHeight(double height){this->height = height;}
    double GetWeight(){return weight;}
    void SetWeight(double weight){this->weight = weight;}

    void SetAll(std::string, double, double);
    Animal(std::string, double, double); //constructor
    Animal(); // overload constructor
    ~Animal(); // destructor
    static int GetNumOfAnimals(){return numOfAnimals;}
    void ToString(); // method that we will be overwriting
};

int Animal::numOfAnimals = 0;
void Animal::SetAll(std::string name, double height, double weight){
    this->name = name;
    this->height = height;
    this->weight = weight;

    Animal::numOfAnimals++;
}

Animal::Animal(std::string name, double height, double weight){
    this->name = name;
    this->height = height;
    this->weight = weight;

    Animal::numOfAnimals++;
}

Animal::Animal(){
    this->name = "";
    this->height = 0;
    this->weight = 0;

    Animal::numOfAnimals++;
}

Animal::~Animal(){
    std::cout << "Animal" << this->name << "destroyed\n" << std::endl;
}

void Animal::ToString(){
    std::cout << this->name << "is" <<
                this->height <<
                " cms tall and " <<
                this->weight <<
                " kgs in weight\n";
}

class Dog: public Animal{
private:
    std::string sound = "Woof";
public:
    void MakeSound(){
        std::cout << "The dog " <<
        this->GetName() << "says" <<
        this->sound << "\n";
    }

    Dog(std::string, double, double, std::string);
    Dog(): Animal(){}; //Animal Default constructor getting initialized
    void ToString();
};

Dog::Dog(std::string name, double height,
 double weight, std::string sound):
 Animal(name, weight, height){
     this->sound = sound;
 }

 void Dog::ToString(){
     std::cout << this-> GetName() << "is" <<
                this->GetHeight() <<
                " cms tall and " <<
                this->GetWeight() <<
                " kgs in weight and says " <<
                this->sound << "\n";
 }

int main()
{
    Animal fred;
    fred.ToString();
    fred.SetWeight(10);
    fred.SetHeight(33);
    fred.SetName("fred");
    fred.ToString();

    Animal tom("Tom", 36, 15);
    tom.ToString();
    Dog spot("Spot", 38, 16, "Woooooof");
    spot.ToString();

    std::cout << "Number of Animansls " <<
        Animal::GetNumOfAnimals() << "\n";
    return 0;
}