# Makefile for Keepaway Players

CC    	= g++
COMPILE = $(CC) $(INCLUDES) ${FLAGS} ${DEFINES}
YACC = bison

SRC_DIR = .
DST_DIR = .
TARGET  = keepaway_player

# ==== INCLUDES =================================================

SOCCER_INCLUDES	= 
INCLUDES	= $(SOCCER_INCLUDES)

# ==== FLAGS ====================================================

# Substitute operating system that is used:
# SunOS_5 linux
OSFLAGS      	= ${OSFLAGS_linux}
CFLAGS_LIBS     = ${CLIBS_linux}
#OSFLAGS        = $(OSFLAGS_SunOS_5)
#CFLAGS_LIBS    = ${CLIBS_SunOS_5}

CFLAGS_DEBUG	= -g
CFLAGS_OPT     	= #-O3 #-static
CFLAGS_WARN     = -Wall
CFLAGS_11       = -std=c++11

OSFLAGS_SunOS_5 = -DSolaris 
OSFLAGS_linux   = -Dlinux -rdynamic

CLIBS_SunOS_5   = 
CLIBS_linux     =

FLAGS           = ${OSFLAGS} ${CFLAGS_OPT} ${CFLAGS_DEBUG} ${CFLAGS_WARN}

# ==== LIBS ====================================================

CLIBS_SunOS_5   = -lsocket -lnsl -lpthread
CLIBS_linux     = -lpthread -ldl

# ==== DEFINES =================================================
#DEFINES	        = -DEC_LOG_ENABLED=1

# ==== SOURCES =================================================

SRCS		=	\
		Connection.cc \
		SenseHandler.cc \
		ActHandler.cc \
		SoccerTypes.cc \
		Objects.cc \
		WorldModel.cc \
		WorldModelHighLevel.cc \
		WorldModelPredict.cc \
		WorldModelUpdate.cc \
		WorldModelKeepaway.cc \
		WorldModelSayMsgTypes.cc \
		ServerSettings.cc \
		PlayerSettings.cc \
		GenericValues.cc \
		Formations.cc \
		Geometry.cc \
		Parse.cc \
		Logger.cc \
		LoggerDraw.cc \

SRCS_PLAYER 	= ${SRCS} \
		BasicPlayer.cc \
		KeepawayPlayer.cc \
		HandCodedAgent.cc \
		tiles2.cc \
		LinearSarsaAgent.cc \
		main.cc

OBJS_PLAYER	= $(SRCS_PLAYER:.cc=.o)

# ==== RULES ==================================================

all::	$(TARGET) demo_extension.so arg_ext.so arg_ext_constant2.so arg_ext_lookback.so arg_ext_argthensarsa.so arg_ext_shared.so arg_ext_shared2.so arg_ext_big.so arg_ext_med.so arg_ext_big_shared.so arg_ext_med_shared.so

	# ^ (niki added many things)

.cc.o:
	$(COMPILE) -c -o $@ $(@F:%.o=%.cc)

# TODO Windows and Mac versions of this.
# TODO Convert the whole build to CMake?
demo_extension.so: DemoExtension.cc
	$(COMPILE) -fPIC -shared -o demo_extension.so DemoExtension.cc

# Niki-written
arg_ext.so: ArgExt.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext.so ArgExt.cc

arg_ext_constant2.so: ArgExtConstant2.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_constant2.so ArgExtConstant2.cc

arg_ext_argthensarsa.so: ArgExtArgThenSarsa.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_argthensarsa.so ArgExtArgThenSarsa.cc

arg_ext_lookback.so: ArgExtLookBack.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_lookback.so ArgExtLookBack.cc

arg_ext_shared.so: ArgSharedBackwards.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_shared.so ArgSharedBackwards.cc

arg_ext_shared2.so: ArgSharedBackwards2.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_shared2.so ArgSharedBackwards2.cc

arg_ext_big.so: ArgExtBig.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_big.so ArgExtBig.cc

arg_ext_big_shared.so: ArgExtBigShared.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_big_shared.so ArgExtBigShared.cc

arg_ext_med.so: ArgExtMed.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_med.so ArgExtMed.cc

arg_ext_med_shared.so: ArgExtMedShared.cc
	$(COMPILE) -fPIC -std=c++11 -shared -o arg_ext_med_shared.so ArgExtMedShared.cc

keepaway_player: $(OBJS_PLAYER)  
	$(COMPILE) -o  $@ $(OBJS_PLAYER) ${CFLAGS_LIBS}

update:
	cvs update

commit:
	cvs commit

clean:
	rm -f $(TARGET) $(OBJS_PLAYER) *core *~ a.out

depend:
	gccmakedep $(DEPENDFLAGS) -s "# DO NOT DELETE THIS LINE" -- $(INCLUDES) -- $(SRCS_PLAYER)

# DO NOT DELETE THIS LINE
