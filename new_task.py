spec = """
================================================================================
BTQ RENDER ENGINE V2.0 - PIXEL-PERFECT VISUAL SPECIFICATION
MACHINE-READABLE FORMAT FOR AUTONOMOUS AGENTS
================================================================================

GLOBAL THEME CONSTANTS:
  BACKGROUND_PRIMARY: #1A1D2E (26, 29, 46)
  BACKGROUND_SECONDARY: #16192B (22, 25, 43)
  TEXT_PRIMARY: #E8E9ED (232, 233, 237)
  TEXT_SECONDARY: #8B8E98 (139, 142, 152)
  BID_GREEN: #00D084 (0, 208, 132)
  ASK_RED: #F6465D (246, 70, 93)
  ACCENT_BLUE: #00C9FF (0, 201, 255)
  ACCENT_YELLOW: #FFD700 (255, 215, 0)
  GRID_LINE: #2B2F42 (43, 47, 66)
  PANEL_BORDER: #3A3F56 (58, 63, 86)
  FONT_MONO: "JetBrains Mono", size=11px, weight=400
  FONT_BOLD: "JetBrains Mono", size=11px, weight=700

================================================================================
COMPONENT 1: ORDER BOOK PANEL (SIZE MODE)
FILE: src/components/orderbook_panel.cpp
DIMENSIONS: width=280px, height=600px
================================================================================

HEADER_BAR:
  position: (0, 0)
  dimensions: (280, 40)
  background: BACKGROUND_SECONDARY
  border_bottom: 1px solid PANEL_BORDER
  
  SYMBOL_DROPDOWN:
    position: (10, 8)
    dimensions: (160, 24)
    background: #0F1218
    border_radius: 4px
    font: FONT_MONO
    text_color: TEXT_PRIMARY
    icon_size: 16x16 (exchange logo)
    icon_position: (14, 12)
    text_position: (34, 12)
    text_sample: "binancef btcusdt"
    dropdown_arrow: (148, 12), size=8x8
    
  DEPTH_SELECTOR:
    position: (175, 8)
    dimensions: (20, 24)
    background: BACKGROUND_PRIMARY
    border: 1px solid PANEL_BORDER
    text: "10"
    text_align: center
    
  CURRENCY_TOGGLE:
    position: (200, 8)
    dimensions: (70, 24)
    background: BACKGROUND_PRIMARY
    border_radius: 4px
    BUTTON_USD:
      position: (200, 8)
      dimensions: (35, 24)
      text: "$ USD"
      icon: dollar_symbol
      active_background: ACCENT_BLUE
    BUTTON_COIN:
      position: (235, 8)
      dimensions: (35, 24)
      text: "🪙 COIN"
      active_background: transparent
      
ORDERBOOK_GRID:
  position: (0, 40)
  dimensions: (280, 560)
  background: BACKGROUND_PRIMARY
  row_height: 20px
  
  ASK_SECTION:
    position: (0, 40)
    rows: 10
    each_row:
      height: 20px
      PRICE_CELL:
        position: (10, y)
        width: 80px
        text_align: right
        text_color: ASK_RED
        font: FONT_MONO
      SIZE_CELL:
        position: (95, y)
        width: 70px
        text_align: right
        text_color: TEXT_PRIMARY
        font: FONT_MONO
      LIQUIDITY_BAR:
        position: (170, y+2)
        height: 16px
        max_width: 100px
        background_gradient: linear(transparent, ASK_RED with alpha=0.3)
        bar_width: (size / max_size) * 100px
        
  MID_PRICE_DISPLAY:
    position: (0, 240)
    dimensions: (280, 40)
    background: #0F1218
    border_top: 1px solid PANEL_BORDER
    border_bottom: 1px solid PANEL_BORDER
    text: "105455.0"
    text_position: center
    font_size: 18px
    font_weight: 700
    text_color: TEXT_PRIMARY
    
  BID_SECTION:
    position: (0, 280)
    rows: 10
    each_row:
      height: 20px
      PRICE_CELL:
        position: (10, y)
        width: 80px
        text_align: right
        text_color: BID_GREEN
        font: FONT_MONO
      SIZE_CELL:
        position: (95, y)
        width: 70px
        text_align: right
        text_color: TEXT_PRIMARY
        font: FONT_MONO
      LIQUIDITY_BAR:
        position: (170, y+2)
        height: 16px
        max_width: 100px
        background_gradient: linear(transparent, BID_GREEN with alpha=0.3)
        bar_width: (size / max_size) * 100px

RENDERING_LOGIC_SIZE_MODE:
  for each level in orderbook.asks[0:10]:
    price_text = format_price(level.price, precision=2)
    size_text = format_size(level.size, precision=2)
    bar_percentage = (level.size / max_visible_size) * 100
    color_intensity = percentile_rank(level.size, trailing_window) / 100.0
    background_alpha = 0.1 + (color_intensity * 0.4)
    render_row(price_text, size_text, bar_percentage, background_alpha, ASK_RED)

================================================================================
COMPONENT 2: ORDER BOOK PANEL (SUM MODE)
FILE: src/components/orderbook_panel.cpp
SAME DIMENSIONS AS COMPONENT 1
================================================================================

DIFFERENCE_FROM_SIZE_MODE:
  SIZE_CELL becomes SUM_CELL
  
  SUM_CELL:
    displays: cumulative_depth[i] = sum(sizes[0..i])
    
  CALCULATION_PSEUDOCODE:
    cumulative_bids = []
    running_sum = 0
    for i, level in enumerate(bids):
      running_sum += level.size
      cumulative_bids[i] = running_sum
      
    cumulative_asks = []
    running_sum = 0
    for i, level in enumerate(asks):
      running_sum += level.size
      cumulative_asks[i] = running_sum
      
  LIQUIDITY_BAR_ADJUSTMENT:
    bar_width = (cumulative_depth[i] / total_book_depth) * 100px
    
  VISUAL_EXAMPLE_ASKS:
    Row 0: 189.85 (sum of top 1 ask)
    Row 1: 165.89 (sum of top 2 asks)
    Row 2: 149.46 (sum of top 3 asks)
    decreasing because we sum from best ask downward
    
  VISUAL_EXAMPLE_BIDS:
    Row 0: 7.38 (sum of top 1 bid)
    Row 1: 21.79 (sum of top 2 bids)
    Row 2: 42.79 (sum of top 3 bids)
    increasing as we accumulate depth

================================================================================
COMPONENT 3: LIQUIDITY HEATMAP (CHART OVERLAY)
FILE: src/analytics/heatmap_renderer.cpp
SHADER: shaders/heatmap_intensity.comp
================================================================================

CHART_VIEWPORT:
  dimensions: (1920, 1080)
  background: BACKGROUND_PRIMARY
  
HEATMAP_LAYER:
  z_index: 1 (below candles, above background)
  opacity: 0.7
  rendering_mode: GPU_COMPUTE_SHADER
  
HEATMAP_DATA_STRUCTURE:
  time_bins: number_of_visible_candles (e.g., 200)
  price_bins: (viewport_height_px / tick_size_px) (e.g., 400 levels)
  data_format: 2D_FLOAT_ARRAY[time_bins][price_bins]
  
SNAPSHOT_LOGIC:
  on_candle_close():
    snapshot = capture_orderbook_L2(symbol, exchange)
    for price_level in snapshot.bids + snapshot.asks:
      price_bin = floor((price_level.price - chart.min_price) / chart.tick_size)
      time_bin = current_candle_index
      heatmap_buffer[time_bin][price_bin] = price_level.size
      
HEATMAP_BINS_HD_MODE:
  tick_aggregation: 1 (every tick is its own bin)
  price_bin_height: 2px
  total_price_bins: viewport_height / 2 = 540 bins
  
HEATMAP_BINS_SD_MODE:
  tick_aggregation: 5 (group 5 ticks together)
  price_bin_height: 10px
  total_price_bins: viewport_height / 10 = 108 bins
  
GPU_SHADER_PIPELINE:
  INPUT_TEXTURE: heatmap_raw (R32F, dimensions: time_bins × price_bins)
  OUTPUT_TEXTURE: heatmap_colored (RGBA8, same dimensions)
  
  SHADER_STAGE_1_FILTERING:
    for each pixel (x, y):
      raw_value = heatmap_raw[x][y]
      if raw_value < params.low_threshold:
        filtered_value = 0.0
      else:
        filtered_value = raw_value
      temp_buffer[x][y] = filtered_value
      
  SHADER_STAGE_2_INTENSITY_MAPPING:
    for each pixel (x, y):
      normalized = clamp(temp_buffer[x][y] / params.peak_threshold, 0.0, 1.0)
      intensity = normalized
      
  SHADER_STAGE_3_BLUR (if mode == SPLAT):
    kernel_3x3 = [
      [0.077, 0.123, 0.077],
      [0.123, 0.195, 0.123],
      [0.077, 0.123, 0.077]
    ]
    for each pixel (x, y):
      blurred = 0.0
      for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
          blurred += temp_buffer[x+dx][y+dy] * kernel_3x3[dx+1][dy+1]
      intensity_buffer[x][y] = blurred
      
  SHADER_STAGE_4_COLORMAP_LOOKUP:
    for each pixel (x, y):
      intensity = intensity_buffer[x][y]
      colormap_u = intensity
      colormap_v = (params.colormap_id + 0.5) / 32.0
      color = texture(colormap_atlas, vec2(colormap_u, colormap_v))
      heatmap_colored[x][y] = color
      
COLORMAP_VIRIDIS_GRADIENT:
  0.0: RGB(68, 1, 84)    # Dark purple
  0.2: RGB(59, 82, 139)  # Blue
  0.4: RGB(33, 145, 140) # Teal
  0.6: RGB(94, 201, 98)  # Green
  0.8: RGB(253, 231, 37) # Yellow
  1.0: RGB(255, 255, 255) # Bright white
  
LIVE_EXTENSION_COLUMN:
  position: (chart_right_edge, 0)
  width: 20px
  height: viewport_height
  update_rate: 100ms
  data_source: current_orderbook_snapshot
  rendering: same as heatmap, but vertical single column
  animation: lerp from previous intensity to new intensity over 200ms
  
UI_CONTROLS_OVERLAY:
  position: (viewport_width / 2 - 200, 10)
  dimensions: (400, 30)
  background: rgba(22, 25, 43, 0.9)
  border_radius: 6px
  
  HD_SD_TOGGLE:
    position: (rel_x=10, rel_y=5)
    dimensions: (60, 20)
    button_hd:
      text: "HD"
      active_background: ACCENT_BLUE
    button_sd:
      text: "SD"
      active_background: transparent
      
  INTENSITY_SLIDER:
    position: (rel_x=80, rel_y=8)
    dimensions: (200, 14)
    track_background: #2B2F42
    fill_background: ACCENT_BLUE
    thumb_size: 12x12
    thumb_color: WHITE
    
  SETTINGS_BUTTON:
    position: (rel_x=290, rel_y=5)
    dimensions: (20, 20)
    icon: gear_icon
    on_click: open_heatmap_settings_panel()

================================================================================
COMPONENT 4: HEATMAP SETTINGS PANEL
FILE: src/ui/heatmap_settings_panel.cpp
================================================================================

SETTINGS_WINDOW:
  position: (viewport_width - 350, 100)
  dimensions: (330, 500)
  background: BACKGROUND_SECONDARY
  border: 1px solid PANEL_BORDER
  border_radius: 8px
  title_bar: "Settings"
  title_bar_height: 30px
  title_bar_background: #0F1218
  
SECTION_STYLE:
  position: (10, 40)
  
  LABEL_HD_TOGGLE:
    position: (10, 40)
    text: "Enable HD Heatmap"
    font: FONT_MONO
    checkbox:
      position: (250, 40)
      size: 16x16
      checked_color: ACCENT_BLUE
      
  DROPDOWN_STYLE:
    position: (10, 70)
    label: "Style"
    label_width: 80px
    dropdown:
      position: (100, 70)
      width: 200px
      height: 24px
      options: ["Classic", "Splat", "Gaussian Splat"]
      
  DROPDOWN_COLORMAP:
    position: (10, 100)
    label: "Colormap"
    label_width: 80px
    dropdown:
      position: (100, 100)
      width: 200px
      height: 24px
      options: ["Viridis", "Plasma", "Inferno", "Turbo", "Hot", "Cool", ...]
      
SECTION_INTENSITY:
  position: (10, 150)
  
  LABEL_INTENSITY:
    text: "INTENSITY"
    font_weight: 700
    
  COLOR_SCALE_PREVIEW:
    position: (10, 180)
    dimensions: (200, 20)
    gradient_fill: linear(
      0%: RGB(68,1,84),
      50%: RGB(33,145,140),
      100%: RGB(253,231,37)
    )
    border: 1px solid PANEL_BORDER
    
  SLIDER_LOW_PEAK:
    position: (10, 210)
    label: "Low/Peak"
    slider_width: 280px
    slider_height: 4px
    range: [0.0, 50000.0]
    thumb_low_position: value_to_pixel(0.100)
    thumb_peak_position: value_to_pixel(40181.152)
    thumb_size: 10x16
    value_labels:
      left: "0.100"
      right: "40181.152"
      
SECTION_MISC:
  position: (10, 280)
  
  CHECKBOX_ZOOM_TOOLTIP:
    position: (10, 310)
    text: "Enable zoom tooltip"
    checked: false
    
  CHECKBOX_EXTEND_HEATMAP:
    position: (10, 340)
    text: "Extend heatmap"
    checked: false
    
SECTION_AGGREGATE:
  position: (10, 380)
  
  LABEL_AGGREGATE:
    text: "AGGREGATE"
    font_weight: 700
    
  EXCHANGE_BUTTONS:
    position: (10, 410)
    button_layout: horizontal_wrap
    button_dimensions: (80, 26)
    button_spacing: 6px
    
    BINANCE:
      text: "Binance"
      active_background: ACCENT_BLUE
      inactive_background: #2B2F42
      
    OKX:
      text: "OKX"
      active_background: ACCENT_BLUE
      
    COINBASE:
      text: "Coinbase"
      active_background: ACCENT_BLUE
      
    BYBIT:
      text: "Bybit"
      active_background: ACCENT_BLUE
      
    HYPERLIQUID:
      text: "Hyperliquid"
      active_background: ACCENT_BLUE

================================================================================
COMPONENT 5: FOOTPRINT CHART (VOLUME HEATMAP)
FILE: src/components/footprint_panel.cpp
SHADER: shaders/footprint_heatmap.comp (reuse heatmap_intensity.comp)
================================================================================

FOOTPRINT_GRID:
  time_axis: X (horizontal)
  price_axis: Y (vertical)
  cell_dimensions: variable based on zoom
    zoom_level_1: 40px width × 4px height
    zoom_level_5: 100px width × 10px height
  
CELL_DATA_STRUCTURE:
  struct FootprintCell {
    double price;
    uint64_t time_start;
    uint64_t time_end;
    double buy_volume;
    double sell_volume;
    double total_volume;
    double delta;
    uint32_t trade_count;
  }
  
RENDERING_MODE_HEATMAP:
  for each visible cell:
    normalized_volume = cell.total_volume / max_candle_volume
    intensity = normalized_volume
    color = lookup_colormap(intensity, VIRIDIS)
    render_rectangle(cell.x, cell.y, cell.width, cell.height, color, alpha=0.8)
    
POC_HIGHLIGHTING:
  for each time_bar (candle):
    max_volume_price = find_price_with_max_volume(time_bar)
    poc_y = price_to_pixel(max_volume_price)
    render_horizontal_line(
      x_start: time_bar.left_edge,
      x_end: time_bar.right_edge,
      y: poc_y,
      color: ACCENT_YELLOW,
      thickness: 2px
    )
    
IMBALANCE_MARKERS:
  for each cell:
    if cell.buy_volume / cell.sell_volume > 3.0:
      render_border(cell, color=BID_GREEN, thickness=2px)
    elif cell.sell_volume / cell.buy_volume > 3.0:
      render_border(cell, color=ASK_RED, thickness=2px)
      
CELL_TOOLTIP_ON_HOVER:
  position: (mouse_x + 10, mouse_y + 10)
  background: rgba(15, 18, 24, 0.95)
  border: 1px solid PANEL_BORDER
  padding: 8px
  text_lines:
    - "Price: {cell.price:.2f}"
    - "Buy: {cell.buy_volume:.2f}"
    - "Sell: {cell.sell_volume:.2f}"
    - "Delta: {cell.delta:.2f}"
    - "Trades: {cell.trade_count}"

================================================================================
COMPONENT 6: TIME & SALES PANEL (TRADES FEED)
FILE: src/components/timeandsales_panel.cpp
================================================================================

PANEL_DIMENSIONS:
  width: 320px
  height: 800px
  
HEADER_BAR:
  position: (0, 0)
  dimensions: (320, 40)
  background: BACKGROUND_SECONDARY
  
  FILTER_INPUT:
    position: (10, 8)
    dimensions: (80, 24)
    placeholder: "Filter"
    text_align: right
    
  RESET_BUTTON:
    position: (100, 8)
    dimensions: (60, 24)
    text: "Reset"
    
  USD_COIN_TOGGLE:
    position: (170, 8)
    dimensions: (70, 24)
    
  SOUND_BUTTON:
    position: (250, 8)
    dimensions: (24, 24)
    icon: speaker_icon
    
TRADES_GRID:
  position: (0, 40)
  dimensions: (320, 760)
  row_height: 18px
  scroll_direction: vertical
  auto_scroll: true (newest at top)
  virtualization: ImGuiListClipper (render only visible rows)
  
  COLUMN_LAYOUT:
    EXCHANGE_LOGO:
      position: (5, row_y + 2)
      dimensions: 14x14
      
    PRICE_COLUMN:
      position: (25, row_y)
      width: 90px
      text_align: right
      text_color: BID_GREEN if is_buy else ASK_RED
      font: FONT_MONO
      
    QUANTITY_COLUMN:
      position: (120, row_y)
      width: 100px
      text_align: right
      font: FONT_MONO
      background_gradient: calculate_intensity_background(trade.size)
      
    TIME_COLUMN:
      position: (225, row_y)
      width: 85px
      text_align: right
      text_color: TEXT_SECONDARY
      font: FONT_MONO
      format: "HH:MM:SS.mmm"
      timezone: UTC
      
BACKGROUND_INTENSITY_CALCULATION:
  trailing_window = deque(maxlen=500)
  trailing_window.append(trade.size)
  
  percentile = calculate_percentile(trade.size, trailing_window)
  base_alpha = 0.05
  max_alpha = 0.5
  alpha = base_alpha + (percentile / 100.0) * (max_alpha - base_alpha)
  
  if trade.is_buy:
    background_color = rgba(BID_GREEN, alpha)
  else:
    background_color = rgba(ASK_RED, alpha)
    
TRADE_FILTERING:
  if filter_input.value > 0:
    display_trade = (trade.size >= filter_input.value)
  else:
    display_trade = true
    
AGGREGATED_MODE:
  if aggregate_enabled:
    trade.exchange_logo = load_icon(trade.exchange_id)
    render_logo(trade.exchange_logo, logo_x, logo_y)

================================================================================
COMPONENT 7: AUDIO FEEDBACK SYSTEM
FILE: src/analytics/audio_engine.cpp
LIBRARY: miniaudio (header-only, included via CMake)
================================================================================

INITIALIZATION:
  ma_engine audio_engine_;
  ma_engine_config config = ma_engine_config_init();
  config.channels = 2;
  config.sampleRate = 48000;
  ma_engine_init(&config, &audio_engine_);
  
  ma_sound buy_sound_;
  ma_sound sell_sound_;
  
  ma_sound_init_from_file(&audio_engine_, "assets/audio/buy_tone.wav", 0, NULL, NULL, &buy_sound_);
  ma_sound_init_from_file(&audio_engine_, "assets/audio/sell_tone.wav", 0, NULL, NULL, &sell_sound_);
  
PITCH_CALCULATION:
  MIN_PITCH = 220.0  // Bass A3
  MAX_PITCH = 880.0  // High A5
  
  log_volume = log10(trade.size)
  log_min = log10(min_recent_trade_size)
  log_max = log10(max_recent_trade_size)
  
  // INVERSE relationship: large trades = low pitch
  normalized = (log_volume - log_min) / (log_max - log_min)
  pitch = MAX_PITCH - (normalized * (MAX_PITCH - MIN_PITCH))
  
PLAYBACK:
  on_trade_event(trade):
    sound = buy_sound_ if trade.is_buy else sell_sound_
    ma_sound_set_pitch(sound, pitch / 440.0)  // 440 Hz = A4 reference
    ma_sound_set_volume(sound, settings.volume)
    ma_sound_start(sound)
    
AUDIO_SETTINGS:
  enable_audio: bool = true
  buy_volume: float = 0.5
  sell_volume: float = 0.5
  base_pitch: float = 440.0
  min_pitch: float = 220.0
  max_pitch: float = 880.0
  note_duration_ms: int = 100
  buy_sound_preset: enum = SINE_WAVE
  sell_sound_preset: enum = SQUARE_WAVE

================================================================================
DATA FLOW ARCHITECTURE
================================================================================

WEBSOCKET_INGESTION:
  exchange_connector.on_trade(raw_trade):
    trade = parse_trade(raw_trade)
    trade_queue.enqueue(trade)
    
  exchange_connector.on_orderbook(raw_book):
    orderbook = parse_orderbook(raw_book)
    orderbook_queue.enqueue(orderbook)
    
WORKER_THREAD_PROCESSING:
  while running:
    trades = trade_queue.dequeue_batch(1000)
    for trade in trades:
      market_data_processor.process_trade(trade)
      cluster_engine.add_trade(trade)
      audio_engine.play_trade_sound(trade.size, trade.is_buy)
      
    orderbooks = orderbook_queue.dequeue_batch(100)
    for book in orderbooks:
      market_data_processor.process_orderbook(book)
      if is_candle_close():
        heatmap_engine.capture_snapshot(book)
        
SUBSCRIPTION_SYSTEM:
  market_data_processor.subscribe(
    symbol_id = BTCUSDT,
    filter = NotificationType.TRADE,
    callback = lambda trade: timesales_panel.add_trade(trade)
  )
  
  market_data_processor.subscribe(
    symbol_id = BTCUSDT,
    filter = NotificationType.ORDERBOOK,
    callback = lambda book: orderbook_panel.update(book)
  )
  
RENDER_LOOP:
  while !should_close():
    // 1. Poll events
    glfwPollEvents()
    
    // 2. Update dirty panels
    for panel in panel_manager.get_dirty_panels():
      panel.update(delta_time)
      
    // 3. Start ImGui frame
    ImGui_ImplVulkan_NewFrame()
    ImGui_ImplGlfw_NewFrame()
    ImGui::NewFrame()
    
    // 4. Render all panels
    for panel in panel_manager.get_all_panels():
      panel.render_gui()
      
    // 5. Render heatmap compute shader
    if heatmap_panel.is_visible():
      heatmap_renderer.dispatch_compute(
        input_buffer = heatmap_engine.get_buffer(),
        params = heatmap_panel.get_shader_params()
      )
      
    // 6. Submit to GPU
    ImGui::Render()
    vulkan_core.record_command_buffer(ImGui::GetDrawData())
    vulkan_core.present_frame()

================================================================================
FILE STRUCTURE (NEW FILES TO CREATE)
================================================================================

include/analytics/heatmap_engine.hpp
  - class HeatmapEngine
  - struct HeatmapSnapshot
  - enum class HeatmapRenderStyle { Classic, Splat, GaussianSplat }
  
include/analytics/heatmap_renderer.hpp
  - class HeatmapRenderer
  - GPU buffer management
  - Compute shader dispatch
  
include/analytics/colormap_manager.hpp
  - class ColormapManager (singleton)
  - Load colormap texture atlas
  - Provide colormap_id lookup
  
include/analytics/footprint_engine.hpp
  - class FootprintEngine
  - struct FootprintCell
  - Build footprint grid from trades
  
include/analytics/audio_engine.hpp
  - class AudioEngine
  - Pitch modulation logic
  - Sound file management
  
include/ui/heatmap_settings_panel.hpp
  - class HeatmapSettingsPanel : public PanelBase
  - ImGui settings rendering
  
src/analytics/heatmap_engine.cpp
  - Snapshot capture logic
  - Circular buffer management
  - Multi-exchange aggregation
  
src/analytics/heatmap_renderer.cpp
  - GPU pipeline setup
  - Compute shader dispatch
  - Texture upload/download
  
src/analytics/colormap_manager.cpp
  - Load 25+ colormaps from assets/
  - Pack into 256×32 texture
  
src/analytics/footprint_engine.cpp
  - Trade aggregation into cells
  - Delta calculation
  - Imbalance detection
  
src/analytics/audio_engine.cpp
  - Miniaudio initialization
  - Pitch calculation
  - Sound playback queue
  
src/ui/heatmap_settings_panel.cpp
  - Render settings UI
  - Handle user input
  - Update shader params
  
shaders/heatmap_intensity.comp
  - GLSL compute shader
  - 4-stage pipeline (filter, intensity, blur, colormap)
  
assets/audio/buy_tone.wav
  - 440 Hz sine wave, 100ms duration
  
assets/audio/sell_tone.wav
  - 440 Hz square wave, 100ms duration
  
assets/colormaps/viridis.png
assets/colormaps/plasma.png
assets/colormaps/turbo.png
  ... (25 total colormap images, 256×1 each)

================================================================================
BUILD INSTRUCTIONS
================================================================================

CMAKE_ADDITIONS:
  # Add miniaudio
  FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(miniaudio)
  target_include_directories(BTQuantTerminal PRIVATE ${miniaudio_SOURCE_DIR})
  
  # Compile compute shaders
  find_program(GLSLC glslc REQUIRED)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/heatmap_intensity.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/heatmap_intensity.comp 
            -o ${CMAKE_BINARY_DIR}/shaders/heatmap_intensity.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/heatmap_intensity.comp
  )

================================================================================
EXACT PIXEL MEASUREMENTS (FOR REFERENCE IMAGE ANALYSIS)
================================================================================

IMAGE_1_ORDERBOOK_SIZE:
  panel_width: 280px
  row_height: 20px
  price_column_width: 80px
  size_column_width: 70px
  bar_column_width: 100px
  header_height: 40px
  mid_price_height: 40px
  font_size: 11px
  line_height: 18px
  
IMAGE_2_HEATMAP:
  viewport_width: 1920px
  viewport_height: 1080px
  heatmap_cell_width: 9.6px (200 candles / 1920px)
  heatmap_cell_height_HD: 2px (540 levels / 1080px)
  heatmap_cell_height_SD: 10px (108 levels / 1080px)
  
IMAGE_3_ORDERBOOK_SUM:
  same dimensions as IMAGE_1
  
IMAGE_4_FOOTPRINT:
  cell_width_zoom_1: 40px
  cell_height_zoom_1: 4px
  poc_line_thickness: 2px
  poc_line_color: #FFD700
  
IMAGE_5_SETTINGS:
  window_width: 330px
  window_height: 500px
  section_spacing: 30px
  label_height: 20px
  input_height: 24px
  button_width: 80px
  button_height: 26px
  
IMAGE_6_TIMESALES:
  panel_width: 320px
  row_height: 18px
  logo_size: 14x14px
  price_column: 90px
  quantity_column: 100px
  time_column: 85px

================================================================================
END OF SPECIFICATION
================================================================================
"""

# Save to file
with open('/tmp/btq_pixel_perfect_spec.txt', 'w') as f:
    f.write(spec)

print("SPECIFICATION WRITTEN TO: /tmp/btq_pixel_perfect_spec.txt")
print(f"LENGTH: {len(spec)} characters")
print("STATUS: READY FOR AGENT CONSUMPTION")
