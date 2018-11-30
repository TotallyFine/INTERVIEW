#include<iostream>
#include<vector>
using namespace std;

int  NumberOf1(int n) {
	// 输出数字中二进制中的1的个数，如果是负数则使用补码 
	int count = 0;
	// 输出数字中二进制中的1的个数，如果是负数则使用补码 
	// 计算机存负数的时候直接就是补码 
 	unsigned int flag = 1;
	while(flag){
 		// flag二进制中1的个数只有一个 
	 	if(flag&n){
  			count ++;
		}
		// flag中的1右移一位 
  		flag = flag << 1;
    }
    return count;
}

int NumberOf1_(int n)
{
    int count = 0;
    while(n){
        count++;  //只要n不为0则其至少有一个1
        // 假如说n=1110 那么n-1就是0111
		// 按位与的结果就是 0110 这样每次进行按位与都能消掉一个1 
        n = n & (n - 1);
    }
    return count;
}

double Power(double base, int exponent) {
	if(exponent<0){
	    base = 1/base;
	    exponent = -exponent;
	}
	double num = base;
    for(int i=0;i<exponent-1;i++){
        num = num * base;
    }
    return num;
    }

//链接：https://www.nowcoder.com/questionTerminal/96bd6684e04a44eb80e6a68efc0ec6c5
class Solution {
public:
    int InversePairs(vector<int> data) {
       int length=data.size();
        if(length<=0)
            return 0;
       //vector<int> copy=new vector<int>[length];
       vector<int> copy;
       for(int i=0;i<length;i++)
           copy.push_back(data[i]);
       long long count=InversePairsCore(data,copy,0,length-1);
       //delete[]copy;
       return count%1000000007;
    }
    long long InversePairsCore(vector<int> &data,vector<int> &copy,int start,int end)
    {
       if(start==end)
          {
            copy[start]=data[start];
            return 0;
          }
       int length=(end-start)/2;
       long long left=InversePairsCore(copy,data,start,start+length);
       long long right=InversePairsCore(copy,data,start+length+1,end); 
        
       int i=start+length;
       int j=end;
       int indexcopy=end;
       long long count=0;
       while(i>=start&&j>=start+length+1)
          {
             if(data[i]>data[j])
                {
                  copy[indexcopy--]=data[i--];
                  count=count+j-start-length;          //count=count+j-(start+length+1)+1;
                }
             else
                {
                  copy[indexcopy--]=data[j--];
                }          
          }
       for(;i>=start;i--)
           copy[indexcopy--]=data[i];
       for(;j>=start+length+1;j--)
           copy[indexcopy--]=data[j];       
       return left+right+count;
    }
};

int add(int num1, int num2){
    // 不使用+ - * / 来加数字
    // 10101001
    // 11010010
    // 先计算两者的异或，这个部分可以算作相加，然后利用与运算&来得到哪些位需要进位 然后左移一位作为进位
    // 一直重复直到没有进位为止
    while(num2!=0){
        int temp = num1^num2;
        num2 = (num1&num2)<<1;
        num1 = temp;
    }
    return num1;
}

int main(){
    //int a = 5;
    //std::cout<<"binary number of 5 contains "<<NumberOf1(5)<<" 1"<<std::endl;
    
    //int exp=-3;
    //std::cout<<Power(2.0, exp)<<std::endl;
    bool k[2]={false};
    std::cout<<k[0]<<k[1]<<std::endl;
}