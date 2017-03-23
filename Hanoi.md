title: Hanoi问题
tags: C++
category: 经典算法
---
##问题描述
有三根相邻的柱子，标号为A,B,C，A柱子上从下到上按金字塔状叠放着n个不同大小的圆盘，要把所有盘子一个一个移动到柱子B上，并且每次移动同一根柱子上都不能出现大盘子在小盘子上方,求移动方法.
<!-- more -->
##解题思路
如果柱子标为ABC,要由A搬至C,在只有一个盘子时,就将它直接搬至C,当有两个盘子,就将B当作辅助柱。如果盘数超过2个,将第三个以下的盘子遮起来,就很简单了,每次处理两个盘子,也就是:A->B、A ->C、B->C这三个步骤,而被遮住的部份,其实就是进入程式的递回处理.事实上,若有n个盘子,则移动完毕所需之次数为2^n - 1.
##代码实现
	#include<iostream>
	using namespace std;

	void hanoi(int n,char A,char B,char C){
		if(n==1)
            cout <<"Move disk "<< n <<" from "<< A << " to " << C <<endl;
        else{
            hanoi(n-1,A,C,B);
            cout << "Mone disk " <<n <<" from "<< A << " to " << C <<endl;
            hanoi(n-1,B,A,C);
        }
    }
    int main(){
        int n;
        cin>>n;
        hanoi(n,'A','B','C');
        return 0;
    }
