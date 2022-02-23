#include <iostream>
#include <execution>
#include <algorithm>
#include <execution>
#include <string>
#include <string.h>
#include <memory.h>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
//#include <boost/algorithm/string/split.hpp>
using namespace std;
namespace common
{
	struct datadef
	{
		int no;
		string type;
		string fmt;
		string label;
		string detail;
		string variable;
		void print()
		{
			cout << type << "," << fmt << "," << label <<"," << detail << "," << variable << "\n";
		}
	};
	vector<datadef> DataDefs;
}


int main(int argc, char *argv[])
{
	std::ifstream ifs(argv[1]);
	if(ifs){
		//cout << "ifs\n";
		string word;
		string line;
		int i =0;
		while(std::getline(ifs,line)){
			//cout << line;
    		stringstream ss(line);
			vector<string> words(6);
			for(int n=0; std::getline(ss,word, ',') && n < words.size(); n++) {
				words[n]= word;
				//cout << "[" << word << "]";
			}
			if(i > 0){
				//cout << "\nw," << words.size() << endl; fflush(stdout);
				common::datadef  datadef_;
				datadef_.no = stoi(words[0]);
				datadef_.type = words[1];
				datadef_.fmt = words[2];
				datadef_.label = words[3];
				datadef_.detail = words[4];
				datadef_.variable = words[5];
				common::DataDefs.push_back(datadef_);
				//cout << words[1];
				if(words[1] == "float"){
					cout << "\tdatap[" << words[0] << "].f32[0] = " << words[5] << ";\n";  
				fflush(stdout);
				}
				else if(words[1] == "int"){
					cout << "\tdatap[" << words[0] << "].u32[0] = " << words[5] << ";\n";  
				}
			}
			i++;
			fflush(stdout);
		}
#if 0
		for(auto & v: common::DataDefs){
			v.print();
		}
#endif
	}

	
	return 0;

}

