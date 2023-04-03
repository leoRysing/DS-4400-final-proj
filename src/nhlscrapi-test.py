
from nhlscrapi.games.game import Game, GameKey, GameType
from nhlscrapi.games.cumstats import Score, ShotCt, Corsi, Fenwick

season = 2014                                    # 2013-2014 season
game_num = 1226                                  #
game_type = GameType.Regular                     # regular season game
game_key = GameKey(season, game_type, game_num)

# define stat types that will be counted as the plays are parsed
cum_stats = {
  'Score': Score(),
  'Shots': ShotCt(),
  'Corsi': Corsi(),
  'Fenwick': Fenwick()
}
game = Game(game_key, cum_stats=cum_stats)

# also http requests and processing are lazy
# accumulators require play by play info so they parse the RTSS PBP
print('Final         : {}'.format(game.cum_stats['Score'].total))
print('Shootout      : {}'.format(game.cum_stats['Score'].shootout.total))
print('Shots         : {}'.format(game.cum_stats['Shots'].total))
print('EV Shot Atts  : {}'.format(game.cum_stats['Corsi'].total))
print('Corsi         : {}'.format(game.cum_stats['Corsi'].share()))
print('FW Shot Atts  : {}'.format(game.cum_stats['Fenwick'].total))
print('Fenwick       : {}'.format(game.cum_stats['Fenwick'].share()))

# http req for roster report
# only parses the sections related to officials and coaches
print('\nRefs          : {}'.format(game.refs))
print('Linesman      : {}'.format(game.linesman))
print('Coaches')
print('  Home        : {}'.format(game.home_coach))
print('  Away        : {}'.format(game.away_coach))

# scrape all remaining reports
game.load_all()
