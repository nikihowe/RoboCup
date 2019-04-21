/**
 * @file 		AF.cpp
 * @class 		AF
 * @brief 		Class encompassing an Argumentation Framework
 * @author 		Mauro Vallati <m.vallati@hud.ac.uk>
 * @copyright	MIT
 */


#include "AF.h"

/**
 * @brief 	Simple constructor
 */
AF::AF()
{
	this->arguments = new SetArguments();
	//this->raw_attacks = map<int, int >();
	//this->attacks = map<Argument *, SetArguments *>();
	//this->attackers = map<Argument *, SetArguments *>();
}


AF::~AF()
{
	// TODO Auto-generated destructor stub
}


string AF::trim(string s)
{
	size_t init = s.find_first_not_of(" ");
	size_t end = s.find_last_not_of(" ");

	if (init == string::npos)
		init = 0;

	if (end == string::npos)
		end = s.size();

	return s.substr(init, end - init + 1);
}

/**
 * @brief 		Method for parsing a ASPARTIX compliant file
 * @details		It uses part of the code from Samer Nofal's, University of Liverpool
 * @param[in] file A string representing the (relative/absolute) path to the ASPARTIX compliant file
 * @retval bool It returns `True` if it can correctly parse the file, `False` otherwise
 */
bool AF::readFile(string file)
{
	string inLine;
	ifstream infile;
	infile.open(file.c_str());
	if (!infile.good())
		return false;

	size_t init = 0;
	do
	{
		getline(infile, inLine);

		if (debug)
			cerr << inLine << endl;

		if (!inLine.empty())
		{
			if ((init = inLine.find("arg")) != string::npos)
			{

				size_t open = inLine.find("(", init + 3);

				if (open == string::npos)
					return false;

				size_t close = inLine.find(")", open + 1);

				if (close == string::npos)
					return false;

				if (debug)
				{
					cerr << inLine.substr(open + 1, close - open - 1)
							<< "; init " << open + 1 << "; close " << close - 1
							<< endl;
					cerr
							<< this->trim(
									inLine.substr(open + 1, close - open - 1))
							<< endl;
				}

				this->arguments->add_Argument(
						new Argument(
								this->trim(
										inLine.substr(open + 1,
												close - open - 1)),
								this->numArgs(), this));

			}
			else if ((init = inLine.find("att")) != string::npos)
			{

				size_t open = inLine.find("(", init + 3);

				if (open == string::npos)
					return false;

				size_t comma = inLine.find(",", open + 1);

				if (comma == string::npos)
					return false;

				size_t close = inLine.find(")", comma + 1);

				if (close == string::npos)
					return false;

				if (debug)
				{
					cerr << this->trim(inLine.substr(open + 1, comma - open - 1)) << endl;
					cerr << this->trim(
							inLine.substr(comma + 1, close - comma - 1)) << endl;
				}

				Argument *source = this->getArgumentByName(
						this->trim(inLine.substr(open + 1, comma - open - 1)));
				Argument *target = this->getArgumentByName(
						this->trim(
								inLine.substr(comma + 1, close - comma - 1)));

				source->add_attacks(target);
				target->add_attackers(source);
			}
		}
	} while (!infile.eof());

	if (debug)
	{
		for (SetArgumentsIterator it = this->arguments->begin();
				it != this->arguments->end(); it++)
			cerr << (*it) << ", ";
		cerr << endl;
	}

//
//	infile >> inLine;
//	while (inLine.find("arg") != string::npos && !infile.eof())
//	{
//		this->arguments->add_Argument(
//				new Argument(inLine.substr(4, inLine.find_last_of(")") - 4),
//						this->numArgs(), this));
//		infile >> inLine;
//	}
//
//	while (!infile.eof())
//	{
//		if (inLine.find("att") != string::npos)
//		{
//			Argument *source = this->getArgumentByName(
//					(inLine.substr(4, inLine.find_last_of(",") - 4)));
//			Argument *target = this->getArgumentByName(
//					(inLine.substr(inLine.find_last_of(",") + 1,
//							inLine.find_last_of(")") - inLine.find_last_of(",")
//									- 1)));
//
//			source->add_attacks(target);
//			target->add_attackers(source);
////			this->raw_attacks.push_back(
////					pair<int, int>(
////							this->argNameToCode(
////									(inLine.substr(4,
////											inLine.find_last_of(",") - 4))),
////							this->argNameToCode(
////									(inLine.substr(inLine.find_last_of(",") + 1,
////											inLine.find_last_of(")")
////													- inLine.find_last_of(",")
////													- 1)))));
//		}
//		infile >> inLine;
//	}
	infile.close();
	return true;
}

/**
 * @brief		This method returns the pointer to the Argument whose name is given as parameter
 * @param[in] name	 The name of the argument
 * @retval Argument* The pointer to the Argument object, or NULL if not found
 */
Argument *AF::getArgumentByName(string name)
{
	return this->arguments->getArgumentByName(name);
}

/**
 * @brief 	Returns the number of arguments
 * @retval int
 */
int AF::numArgs()
{
	return this->arguments->cardinality();
}


/**
 * @brief		This method returns the pointer to the Argument whose identification number is given as parameter
 * @param[in] num	 The name of the argument
 * @retval Argument* The pointer to the Argument object, or NULL if not found
 */
Argument *AF::getArgumentByNumber(int num)
{
	return this->arguments->getArgumentByNumber(num);
}

/**
 * @brief 	Begin of the iterator for the set of arguments
 * @retval SetArgumentsIterator An iterator pointing at the first of the elements of the set of arguments
 */
SetArgumentsIterator AF::begin()
{
	return this->arguments->begin();
}

/**
 * @brief 	End of the iterator for the set of arguments
 * @retval SetArgumentsIterator An iterator pointing at the last of the elements of the set of arguments
 */
SetArgumentsIterator AF::end()
{
	return this->arguments->end();
}

/**
 * @brief	Returns a pointer to the set of arguments
 * @retval SetArguments*
 */
SetArguments *AF::get_arguments() const
{
	return this->arguments;
}
