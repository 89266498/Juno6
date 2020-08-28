
CREATE TABLE `AI_instruction` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_time` datetime DEFAULT NULL,
  `creator` varchar(128) DEFAULT NULL,
  `start_time` varchar(128) DEFAULT NULL,
  `end_time` varchar(128) DEFAULT NULL,
  `control_variate` varchar(128) DEFAULT NULL,
  `target_value` varchar(128) DEFAULT NULL,
  `rate_progress` varchar(128) DEFAULT NULL,
  `schedule_score` varchar(128) DEFAULT NULL,
  `relate_variate` varchar(128) DEFAULT NULL,
  `expectation` varchar(128) DEFAULT NULL,
  `expectation_score` varchar(128) DEFAULT NULL,
  `ai_mark` varchar(128) DEFAULT NULL,
  `comment` varchar(512) DEFAULT NULL,
   PRIMARY KEY (`id`),
  UNIQUE KEY `AI_instruction_id_uindex` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


CREATE TABLE `AI_mark_score` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_time` datetime DEFAULT NULL,
  `total_score` varchar(128) DEFAULT NULL,
  `total_mark` varchar(128) DEFAULT NULL,
  `emission_score` varchar(128) DEFAULT NULL,
  `op_cost_score` varchar(128) DEFAULT NULL,
  `instruction_score` varchar(128) DEFAULT NULL,
  `equipment_score` varchar(128) DEFAULT NULL,
  `data_quality` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `AI_mark_score_id_uindex` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `AI_predict` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_time` datetime DEFAULT NULL,
  `predict_content` varchar(512) DEFAULT NULL,
  `predit_diagram` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `AI_predict_id_uindex` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `AI_analysis` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_time` datetime DEFAULT NULL,
  `analysis_content` varchar(512) DEFAULT NULL,
  `predit_clue_list` varchar(512) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `AI_analysis_id_uindex` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `AI_cost_computer` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_time` datetime DEFAULT NULL,
  `energy_cost` varchar(128) DEFAULT NULL,
  `energy_cost_period` varchar(128) DEFAULT NULL,
  `pam_cost` varchar(128) DEFAULT NULL,
  `pam_cost_period` varchar(128) DEFAULT NULL,
  `pac_cost` varchar(128) DEFAULT NULL,
  `pac_cost_period` varchar(128) DEFAULT NULL,
  `carbon_cost` varchar(128) DEFAULT NULL,
  `carbon_cost_period` varchar(128) DEFAULT NULL,
  `lox_cost` varchar(128) DEFAULT NULL,
  `lox_cost_period` varchar(128) DEFAULT NULL,
  `sludge_cost` varchar(128) DEFAULT NULL,
  `sludge_cost_period` varchar(128) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `AI_cost_computer_id_uindex` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


