#include<iostream>
using namespace std;

class hashNode {
private:
    int key;
    int value;
    hashNode *next;
public:
    hashNode(int key, int value) {
        this->key = key;
        this->value = value;
        this->next = NULL;
    }
    int getKey() {
        return key;
    }
    int getValue() {
        return value;
    }
    void setValue(int value) {
        this->value = value;
    }
    hashNode *getNext() {
        return next;
    }
    void setNext(hashNode *next) {
        this->next = next;
    }
};

class hashMap {
private:
    int BUCKET;
    hashNode **table;
    
}