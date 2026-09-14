package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	demoinfocs "github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
)

const (
	parserVersion  = "demoinfocs-golang/v3.3.0"
	rawSchema      = "RawTickV1"
	allowedMapName = "de_mirage"
	demoRoot       = `E:\Demo`
	flushRows      = 2048
)

type rawFrame struct {
	serverTick int32
	demoTime   float64
	deltaTime  float32
	players    []map[string]any
	events     []map[string]any
}

type arrowSink struct {
	file   *os.File
	writer *ipc.FileWriter
	builder *array.RecordBuilder
	rows   int
}

func rawSchemaV1(purpose, mapName string) *arrow.Schema {
	metadata := arrow.NewMetadata(
		[]string{"schema", "purpose", "map_name", "parser_version"},
		[]string{rawSchema, purpose, mapName, parserVersion},
	)
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "server_tick", Type: arrow.PrimitiveTypes.Int32},
			{Name: "demo_time_s", Type: arrow.PrimitiveTypes.Float64},
			{Name: "delta_time_s", Type: arrow.PrimitiveTypes.Float32},
			{Name: "players_json", Type: arrow.BinaryTypes.String},
			{Name: "events_json", Type: arrow.BinaryTypes.String},
		},
		&metadata,
	)
}

func newArrowSink(output, purpose, mapName string) (*arrowSink, error) {
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		return nil, fmt.Errorf("create output directory: %w", err)
	}
	file, err := os.Create(output)
	if err != nil {
		return nil, fmt.Errorf("create output: %w", err)
	}
	schema := rawSchemaV1(purpose, mapName)
	writer, err := ipc.NewFileWriter(file, ipc.WithSchema(schema))
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("create Arrow writer: %w", err)
	}
	return &arrowSink{
		file:    file,
		writer:  writer,
		builder: array.NewRecordBuilder(memory.DefaultAllocator, schema),
	}, nil
}

func (sink *arrowSink) append(frame rawFrame) error {
	if frame.players == nil {
		frame.players = []map[string]any{}
	}
	if frame.events == nil {
		frame.events = []map[string]any{}
	}
	sink.builder.Field(0).(*array.Int32Builder).Append(frame.serverTick)
	sink.builder.Field(1).(*array.Float64Builder).Append(frame.demoTime)
	sink.builder.Field(2).(*array.Float32Builder).Append(frame.deltaTime)
	players, err := json.Marshal(frame.players)
	if err != nil {
		return fmt.Errorf("encode players at tick %d: %w", frame.serverTick, err)
	}
	eventsJSON, err := json.Marshal(frame.events)
	if err != nil {
		return fmt.Errorf("encode events at tick %d: %w", frame.serverTick, err)
	}
	sink.builder.Field(3).(*array.StringBuilder).Append(string(players))
	sink.builder.Field(4).(*array.StringBuilder).Append(string(eventsJSON))
	sink.rows++
	if sink.rows >= flushRows {
		return sink.flush()
	}
	return nil
}

func (sink *arrowSink) flush() error {
	if sink.rows == 0 {
		return nil
	}
	record := sink.builder.NewRecord()
	defer record.Release()
	if err := sink.writer.Write(record); err != nil {
		return fmt.Errorf("write Arrow record batch: %w", err)
	}
	sink.rows = 0
	return nil
}

func (sink *arrowSink) close() error {
	var firstErr error
	if err := sink.flush(); err != nil {
		firstErr = err
	}
	sink.builder.Release()
	if err := sink.writer.Close(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("close Arrow writer: %w", err)
	}
	if err := sink.file.Close(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("close Arrow file: %w", err)
	}
	return firstErr
}

func playerRef(player *common.Player) map[string]any {
	if player == nil {
		return nil
	}
	return map[string]any{
		"entity_id":  player.EntityID,
		"steam_id32": player.SteamID32(),
		"name":       player.Name,
	}
}

func equipmentName(equipment *common.Equipment) string {
	if equipment == nil {
		return ""
	}
	return equipment.String()
}

func vectorValues(x, y, z float64) []float64 {
	return []float64{x, y, z}
}

func playerSnapshot(player *common.Player) map[string]any {
	position := player.Position()
	eyePosition := player.PositionEyes()
	velocity := player.Velocity()
	activeWeapon := player.ActiveWeapon()
	weapons := make([]map[string]any, 0, len(player.Weapons()))
	for _, weapon := range player.Weapons() {
		if weapon == nil {
			continue
		}
		weapons = append(weapons, map[string]any{
			"name":           weapon.String(),
			"unique_id":      weapon.UniqueID(),
			"ammo_in_mag":    weapon.AmmoInMagazine(),
			"ammo_reserve":   weapon.AmmoReserve(),
			"ammo_type":      weapon.AmmoType(),
			"equipment_class": fmt.Sprint(weapon.Class()),
		})
	}
	sort.Slice(weapons, func(i, j int) bool {
		return weapons[i]["unique_id"].(int64) < weapons[j]["unique_id"].(int64)
	})
	return map[string]any{
		"entity_id":          player.EntityID,
		"steam_id32":         player.SteamID32(),
		"name":               player.Name,
		"team":               int(player.Team),
		"is_bot":             player.IsBot,
		"is_connected":       player.IsConnected,
		"is_alive":           player.IsAlive(),
		"position":           vectorValues(position.X, position.Y, position.Z),
		"position_eyes":      vectorValues(eyePosition.X, eyePosition.Y, eyePosition.Z),
		"velocity":           vectorValues(velocity.X, velocity.Y, velocity.Z),
		"view_yaw_deg":       player.ViewDirectionX(),
		"view_pitch_deg":     player.ViewDirectionY(),
		"health":             player.Health(),
		"armor":              player.Armor(),
		"money":              player.Money(),
		"is_ducking":         player.IsDucking(),
		"is_walking":         player.IsWalking(),
		"is_scoped":          player.IsScoped(),
		"is_airborne":        player.IsAirborne(),
		"is_blinded":         player.IsBlinded(),
		"flash_remaining_s":  player.FlashDurationTimeRemaining().Seconds(),
		"has_helmet":         player.HasHelmet(),
		"has_defuse_kit":     player.HasDefuseKit(),
		"is_in_bomb_zone":    player.IsInBombZone(),
		"is_in_buy_zone":     player.IsInBuyZone(),
		"is_defusing":        player.IsDefusing,
		"is_planting":        player.IsPlanting,
		"is_reloading":       player.IsReloading,
		"active_weapon":      equipmentName(activeWeapon),
		"active_weapon_ammo": activeAmmo(activeWeapon),
		"weapons":            weapons,
		"last_place_name":    player.LastPlaceName(),
	}
}

func activeAmmo(weapon *common.Equipment) map[string]int {
	if weapon == nil {
		return map[string]int{"in_magazine": 0, "reserve": 0}
	}
	return map[string]int{
		"in_magazine": weapon.AmmoInMagazine(),
		"reserve":     weapon.AmmoReserve(),
	}
}

func projectileSnapshot(projectile *common.GrenadeProjectile) map[string]any {
	if projectile == nil {
		return nil
	}
	position := projectile.Position()
	velocity := projectile.Velocity()
	return map[string]any{
		"unique_id":       projectile.UniqueID(),
		"weapon":          equipmentName(projectile.WeaponInstance),
		"thrower":         playerRef(projectile.Thrower),
		"owner":           playerRef(projectile.Owner),
		"position":        vectorValues(position.X, position.Y, position.Z),
		"velocity":        vectorValues(velocity.X, velocity.Y, velocity.Z),
		"trajectory_len":  len(projectile.Trajectory),
	}
}

func appendEvent(eventsList *[]map[string]any, event map[string]any) {
	*eventsList = append(*eventsList, event)
}

func recordGrenadeEvent(pending *[]map[string]any, name string, event events.GrenadeEventIf) {
	base := event.Base()
	appendEvent(pending, map[string]any{
		"type":      name,
		"grenade":   base.GrenadeType.String(),
		"thrower":   playerRef(base.Thrower),
		"entity_id": base.GrenadeEntityID,
		"position":  vectorValues(base.Position.X, base.Position.Y, base.Position.Z),
	})
}

func registerEventHandlers(parser demoinfocs.Parser, pending *[]map[string]any) {
	parser.RegisterEventHandler(func(event events.WeaponFire) {
		appendEvent(pending, map[string]any{
			"type":    "weapon_fire",
			"shooter": playerRef(event.Shooter),
			"weapon":  equipmentName(event.Weapon),
		})
	})
	parser.RegisterEventHandler(func(event events.WeaponReload) {
		appendEvent(pending, map[string]any{
			"type":   "weapon_reload",
			"player": playerRef(event.Player),
		})
	})
	parser.RegisterEventHandler(func(event events.PlayerHurt) {
		appendEvent(pending, map[string]any{
			"type":               "player_hurt",
			"player":             playerRef(event.Player),
			"attacker":           playerRef(event.Attacker),
			"weapon":             equipmentName(event.Weapon),
			"health":             event.Health,
			"armor":              event.Armor,
			"health_damage":      event.HealthDamage,
			"armor_damage":       event.ArmorDamage,
			"health_damage_taken": event.HealthDamageTaken,
			"armor_damage_taken":  event.ArmorDamageTaken,
			"hit_group":          int(event.HitGroup),
		})
	})
	parser.RegisterEventHandler(func(event events.PlayerFlashed) {
		appendEvent(pending, map[string]any{
			"type":          "player_flashed",
			"player":        playerRef(event.Player),
			"attacker":      playerRef(event.Attacker),
			"flash_duration": event.FlashDuration().Seconds(),
		})
	})
	parser.RegisterEventHandler(func(event events.Kill) {
		appendEvent(pending, map[string]any{
			"type":              "kill",
			"killer":            playerRef(event.Killer),
			"victim":            playerRef(event.Victim),
			"assister":          playerRef(event.Assister),
			"weapon":            equipmentName(event.Weapon),
			"penetrated_objects": event.PenetratedObjects,
			"is_headshot":       event.IsHeadshot,
			"assisted_flash":    event.AssistedFlash,
			"attacker_blind":    event.AttackerBlind,
			"no_scope":          event.NoScope,
			"through_smoke":     event.ThroughSmoke,
			"distance":          event.Distance,
		})
	})
	parser.RegisterEventHandler(func(event events.GrenadeProjectileThrow) {
		appendEvent(pending, map[string]any{
			"type":       "grenade_projectile_throw",
			"projectile": projectileSnapshot(event.Projectile),
		})
	})
	parser.RegisterEventHandler(func(event events.GrenadeProjectileBounce) {
		appendEvent(pending, map[string]any{
			"type":       "grenade_projectile_bounce",
			"projectile": projectileSnapshot(event.Projectile),
			"bounce_nr":  event.BounceNr,
		})
	})
	parser.RegisterEventHandler(func(event events.GrenadeProjectileDestroy) {
		appendEvent(pending, map[string]any{
			"type":       "grenade_projectile_destroy",
			"projectile": projectileSnapshot(event.Projectile),
		})
	})
	parser.RegisterEventHandler(func(event events.Footstep) {
		appendEvent(pending, map[string]any{
			"type":   "footstep",
			"player": playerRef(event.Player),
		})
	})
	parser.RegisterEventHandler(func(event events.PlayerJump) {
		appendEvent(pending, map[string]any{
			"type":   "player_jump",
			"player": playerRef(event.Player),
		})
	})
	parser.RegisterEventHandler(func(event events.RoundStart) {
		appendEvent(pending, map[string]any{"type": "round_start"})
	})
	parser.RegisterEventHandler(func(event events.RoundEnd) {
		appendEvent(pending, map[string]any{
			"type":    "round_end",
			"message": event.Message,
			"reason":  int(event.Reason),
			"winner":  int(event.Winner),
		})
	})
	parser.RegisterEventHandler(func(event events.RoundFreezetimeEnd) {
		appendEvent(pending, map[string]any{"type": "round_freezetime_end"})
	})
	parser.RegisterEventHandler(func(event events.TeamSideSwitch) {
		appendEvent(pending, map[string]any{"type": "team_side_switch"})
	})
	parser.RegisterEventHandler(func(event events.SmokeStart) {
		recordGrenadeEvent(pending, "smoke_start", event)
	})
	parser.RegisterEventHandler(func(event events.SmokeExpired) {
		recordGrenadeEvent(pending, "smoke_expired", event)
	})
	parser.RegisterEventHandler(func(event events.FlashExplode) {
		recordGrenadeEvent(pending, "flash_explode", event)
	})
	parser.RegisterEventHandler(func(event events.FireGrenadeStart) {
		recordGrenadeEvent(pending, "fire_grenade_start", event)
	})
	parser.RegisterEventHandler(func(event events.FireGrenadeExpired) {
		recordGrenadeEvent(pending, "fire_grenade_expired", event)
	})
	parser.RegisterEventHandler(func(event events.HeExplode) {
		recordGrenadeEvent(pending, "he_explode", event)
	})
	parser.RegisterEventHandler(func(event events.DecoyStart) {
		recordGrenadeEvent(pending, "decoy_start", event)
	})
	parser.RegisterEventHandler(func(event events.DecoyExpired) {
		recordGrenadeEvent(pending, "decoy_expired", event)
	})
}

func capturePlayers(parser demoinfocs.Parser) []map[string]any {
	players := parser.GameState().Participants().Playing()
	snapshots := make([]map[string]any, 0, len(players))
	for _, player := range players {
		if player == nil {
			continue
		}
		snapshots = append(snapshots, playerSnapshot(player))
	}
	sort.Slice(snapshots, func(i, j int) bool {
		return snapshots[i]["entity_id"].(int) < snapshots[j]["entity_id"].(int)
	})
	return snapshots
}

func parseDemo(input, output, purpose string) (common.DemoHeader, error) {
	if purpose != "test_only" && purpose != "production" {
		return common.DemoHeader{}, errors.New("purpose must be test_only or production")
	}
	if isUnderDemoRoot(input) && purpose == "production" {
		return common.DemoHeader{}, errors.New("E:\\Demo inputs are always test_only")
	}
	file, err := os.Open(input)
	if err != nil {
		return common.DemoHeader{}, fmt.Errorf("open demo: %w", err)
	}
	defer file.Close()
	parser := demoinfocs.NewParser(file)
	defer parser.Close()
	header, err := parser.ParseHeader()
	if err != nil {
		return common.DemoHeader{}, fmt.Errorf("parse demo header: %w", err)
	}
	if header.Filestamp != "HL2DEMO" {
		return common.DemoHeader{}, fmt.Errorf("unexpected demo stamp %q", header.Filestamp)
	}
	if header.MapName != allowedMapName {
		return common.DemoHeader{}, fmt.Errorf("only %s is supported, got %s", allowedMapName, header.MapName)
	}
	sink, err := newArrowSink(output, purpose, header.MapName)
	if err != nil {
		return common.DemoHeader{}, err
	}
	closed := false
	closeSink := func() error {
		if closed {
			return nil
		}
		closed = true
		return sink.close()
	}
	defer func() { _ = closeSink() }()
	var pending []map[string]any
	var sinkErr error
	lastTick := -1
	lastTime := -1.0
	registerEventHandlers(parser, &pending)
	parser.RegisterEventHandler(func(event events.FrameDone) {
		if sinkErr != nil {
			return
		}
		tick := parser.GameState().IngameTick()
		now := parser.CurrentTime().Seconds()
		if tick < 0 || tick == lastTick {
			pending = nil
			return
		}
		if lastTime >= 0 && now < lastTime {
			sinkErr = fmt.Errorf("demo time moved backwards at tick %d", tick)
			parser.Cancel()
			return
		}
		delta := 0.0
		if lastTime >= 0 {
			delta = now - lastTime
		}
		if err := sink.append(rawFrame{
			serverTick: int32(tick),
			demoTime:   now,
			deltaTime:  float32(delta),
			players:    capturePlayers(parser),
			events:     pending,
		}); err != nil {
			sinkErr = err
			parser.Cancel()
			return
		}
		lastTick = tick
		lastTime = now
		pending = nil
	})
	if err := parser.ParseToEnd(); err != nil && sinkErr == nil {
		return common.DemoHeader{}, fmt.Errorf("parse demo: %w", err)
	}
	if sinkErr != nil {
		return common.DemoHeader{}, sinkErr
	}
	if err := closeSink(); err != nil {
		return common.DemoHeader{}, err
	}
	return header, nil
}

func isUnderDemoRoot(path string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	root, err := filepath.Abs(demoRoot)
	if err != nil {
		return false
	}
	abs = filepath.Clean(abs)
	root = filepath.Clean(root)
	if strings.EqualFold(abs, root) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(abs), strings.ToLower(root+string(os.PathSeparator)))
}

func run() error {
	input := flag.String("input", "", "Source 1 GOTV demo path")
	output := flag.String("output", "", "Arrow IPC output path")
	purpose := flag.String("purpose", "", "test_only or production")
	flag.Parse()
	if *input == "" || *output == "" || *purpose == "" {
		return errors.New("--input, --output and --purpose are required")
	}
	header, err := parseDemo(*input, *output, *purpose)
	if err != nil {
		return err
	}
	info, err := os.Stat(*output)
	if err != nil {
		return fmt.Errorf("stat Arrow output: %w", err)
	}
	fmt.Printf("map=%s ticks=%d frames=%d output=%s bytes=%d\n", header.MapName, header.PlaybackTicks, header.PlaybackFrames, *output, info.Size())
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
