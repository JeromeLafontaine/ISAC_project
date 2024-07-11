//
// Copyright 2010-2012,2014 Ettus Research LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <stdlib.h>
//#include <unistd.h>

static const size_t wave_table_len = 64; // this is the number of complex samples for one period of the wave!
// in order to load a custom wave, the necessary for loop counts until wave_table_len*2, since there are in-phase and quadrature components for each complex sample.
 
class wave_table_class{
public:
    wave_table_class(const std::string &wave_type, const float ampl): 
        _wave_table(wave_table_len)
    {
	std::vector<double> real_wave_table(wave_table_len*2);
	if (wave_type == "CONST"){
            for (size_t i = 0; i < wave_table_len; i++)
                real_wave_table[i] = 1.0;
        }
        else if (wave_type == "SQUARE"){
            for (size_t i = 0; i < wave_table_len; i++)
                real_wave_table[i] = (i < wave_table_len/2)? 0.0 : 1.0;
        }
        else if (wave_type == "RAMP"){
            for (size_t i = 0; i < wave_table_len; i++)
                real_wave_table[i] = 2.0*i/(wave_table_len-1) - 1.0;
        }
        else if (wave_type == "SINE"){
            static const double tau = 2*std::acos(-1.0);
            for (size_t i = 0; i < wave_table_len; i++){
                real_wave_table[i] = std::sin((tau*i)/wave_table_len);
		}
        }
	else if (wave_type == "CUSTOM"){
		std::ifstream infile("custom_wave.txt", std::ifstream::in);
		if(infile.is_open()){
			std::string tempString;
			int lenCounter = 0;
			std::string::size_type size;
			while(std::getline(infile, tempString)){
				real_wave_table[lenCounter] = std::stod(tempString, &size);
				lenCounter++;
			}
		}
		else
			std::runtime_error("couldnot load");
	}
        else throw std::runtime_error("unknown waveform type: " + wave_type);

        //compute i and q pairs with 90% offset and scale to amplitude
	    
	if (wave_type == "CUSTOM"){
       		for (size_t i = 0; i < wave_table_len; i++){
            		_wave_table[i] = std::complex<float>(ampl*real_wave_table[2*i], ampl*real_wave_table[2*i+1]);
        	}
	}    
	else{
		for (size_t i = 0; i < wave_table_len; i++){
            	const size_t q = (i+(3*wave_table_len)/4)%wave_table_len;
            	_wave_table[i] = std::complex<float>(ampl*real_wave_table[i], ampl*real_wave_table[q]);
        	}
	}
    }

    inline std::complex<float> operator()(const size_t index) const{
        return _wave_table[index % wave_table_len];
    }

private:
    std::vector<std::complex<float> > _wave_table;
    std::streampos size;
    char * memblock;
};

